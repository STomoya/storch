"""Distributed training helper."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from storch.distributed import utils
from storch.distributed._state import DistributedState
from storch.distributed.factory import (
    DistributedDataParallelFactory,
    FullyShardedDataParallelFactory,
    NoParallelFactory,
    simple,
)
from storch.torchops import convert_outputs_to_fp32
from storch.utils import version


class DistributedHelper:
    """Helper class for distributed training.

    Examples
    --------
        >>> # When running with torchrun launcher, nothing needs to be set.
        >>> disthelper = DistributedHelper(world_size, rank) # arguments are ignored.
        >>> # When lauching with torch.multiprocessing.spawn
        >>> os.environ['MASTER_ADDR'] = '127.0.0.1'
        >>> os.environ['MASTER_PORT'] = '29500'
        >>> #   in main function
        >>> disthelper = DistributedHelper(world_size=world_size, rank=rank)

    """

    def __init__(self, **kwargs) -> None:  # noqa: D417
        """NeST.

        Args:
        ----
            world_size (int, optional): world size. ignored when {torchrun,CPU,single GPU}. Default: 1.
            rank (int, optional): rank. ignored when {torchrun,CPU,single GPU}. Default: 0.

        """
        self._state = DistributedState(**kwargs)
        self.backend = self._state.backend
        self.world_size = self._state.num_processes
        self.rank = self._state.process_index
        self.local_rank = self._state.local_process_index
        self.device = self._state.device

        self._mode = None
        self._factories = []

    def __repr__(self) -> str:  # noqa: D105
        return self._state.__repr__()

    @staticmethod
    def is_available() -> bool:
        """Is distributed module available."""
        return dist.is_available()

    @staticmethod
    def is_initialized() -> bool:
        """Is distributed process group initialized."""
        return dist.is_initialized()

    @staticmethod
    def is_torchrun() -> bool:
        """Was python launched using `torchrun` command."""
        return dist.is_torchelastic_launched()

    def is_primary(self) -> bool:
        """Is the process the primary process."""
        return self._state.is_main_process

    # gather functions.
    # these functions work on both single/multi GPU without any specifications.

    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather all tensors, scattered to multiple devices, to a tensor concatenated on the 0th dimension.

        Args:
        ----
            tensor (torch.Tensor): the tensor to gather

        Returns:
        -------
            torch.Tensor: gathered tensor

        """
        output_tensor = utils.gather(tensor, dst=None, into_tensor=True)
        return output_tensor

    def gather_any(self, val: Any) -> torch.Tensor:
        """Gather that supports float and int input.

        Args:
        ----
            val (Any): value to gather.

        Returns:
        -------
            torch.Tensor: gathered values as an tensor.

        """
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device=self.device)
        val = self.gather_tensor(val)
        return val

    def reduce_tensor(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
        """Reduce a tensor, scattered on multiple GPUs, with a certain operator.

        Args:
        ----
            tensor (torch.Tensor): the tensor to reduce
            op (ReduceOp, optional): the operator. Default: ReduceOp.SUM.

        Returns:
        -------
            torch.Tensor: the reduced tensor

        """
        tensor = utils.reduce(tensor, dst=None, op=op)
        return tensor

    def reduce_any(self, val: Any, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
        """Reduce that supports float and int inputs.

        Args:
        ----
            val (Any): value to reduce
            op (ReduceOp, optional): operator. Default: ReduceOp.SUM.

        Returns:
        -------
            torch.Tensor: the reduced tensor

        """
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device=self.device)
        val = self.reduce_tensor(val, op=op)
        return val

    # aliases
    gather = gather_any
    reduce = reduce_any

    def reduce_sum(self, val: Any) -> torch.Tensor:
        """Reduce distributed tensors by summation."""
        return self.reduce_any(val)

    def reduce_mean(self, val: Any) -> torch.Tensor:
        """Reduce distributed tensors by taking the average."""
        return self.reduce_any(val, ReduceOp.AVG)

    def reduce_prod(self, val: Any) -> torch.Tensor:
        """Reduce distributed tensors by production."""
        return self.reduce_any(val, ReduceOp.PRODUCT)

    def reduce_min(self, val: Any) -> torch.Tensor:
        """Reduce distributed tensors by taking the minmum."""
        return self.reduce_any(val, ReduceOp.MIN)

    def reduce_max(self, val: Any) -> torch.Tensor:
        """Reduce distributed tensors by taking the maximum."""
        return self.reduce_any(val, ReduceOp.MAX)

    @staticmethod
    def barrier():
        """Syncronize all processes."""
        utils.wait_for_everyone()

    @staticmethod
    def wait_for_all_processes():
        """torch.distributed.barrier with a recognizable name."""
        utils.wait_for_everyone()

    def get_parallel_mode(self, mode: str | None = None) -> str:
        """Get parallelizm strategy.

        The user should always call this function to get the strategy, because this function forces the `_mode` attr to
        be set only once.

        If mode is not set, initialize the `_mode` attribute using the given `mode` argument. if `mode is None`, we
        fallback to `'ddp'` which is more stable than FSDP. If `_mode` attr is already set, returns the value of
        `_mode`.

        Args:
        ----
            mode (str, optional): the parallelizm mode. Default: None.

        Returns:
        -------
            str: a string representing the parallelizm strategy.

        """
        if self._mode is None:
            if mode is None:
                mode = 'ddp'
                if self.is_primary():
                    print('Data parallelizm default to "ddp".')

            assert mode.lower() in ['ddp', 'fsdp', 'none']
            self._mode = mode

        return self._mode

    def prepare_module(
        self,
        *modules: nn.Module,
        mode: str | None = None,
        mixed_precision: bool = False,
        compile: bool | str | dict | None = None,
        return_intermediates: bool = False,
    ) -> tuple[nn.Module] | nn.Module:
        """Prepare the input modules with mode "mode".

        - If the device is a cuda device the module is first set to the device, then wrapped.
        - "mixed_precision" is enabled by wrapping the forward function with torch.cuda.amp.autocast,
            and storch.torchops.convert_outputc_to_fp32. No need to use autocast contextmanager.

        Args:
        ----
            *modules (nn.Module): modules to wrap for distributed training.
            mode (str, optional): data parallel mode. one of {ddp,fsdp} for distributed settings,
                any other for single GPU or CPU. Default: None.
            mixed_precision (bool, optional): use mixed precision. Default: False.
            compile (bool | str | dict, optional): compile the model via torch.compile.
                - if True, uses torch.compile without any arguments.
                - if dict, uses torch.compile and use dict assuming containing arguments (e.g. "mode")
                - if str,  uses torch.compile and input str as "mode" argument.
                Default: None.
            return_intermediates: Return itermediate models. Default: False.

        Raises:
        ------
            Exception: unknown data parallel mode.

        Returns:
        -------
            tuple[nn.Module]|nn.Module: the wrapped modules in the same order as input.

        """
        mode = self.get_parallel_mode(mode)

        original_modules = []
        parallel_modules = []
        compiled_modules = []
        for module in modules:
            # confirm send module to device before wrapping the model
            module.to(self.device)

            original_modules.append(module)

            # wrap the model. see storch.distributed.factory.
            if version.is_dist_state_dict_available():
                wrapped_module = self._wrap_module(module, mode, mixed_precision, compile)
            else:
                wrapped_module = self._wrap_module_legacy(module, mode, mixed_precision, compile)

            # wrap forward with autocast if enabled.
            if mixed_precision and self.device.type == 'cuda':
                wrapped_module._original_forward = wrapped_module.forward
                dtype = torch.float16
                if mixed_precision == 'bf16':
                    dtype = torch.bfloat16
                forward = torch.cuda.amp.autocast(dtype=dtype)(wrapped_module.forward)
                wrapped_module.forward = convert_outputs_to_fp32(forward)

            parallel_modules.append(wrapped_module)

            # compile model
            compiled_module = wrapped_module
            if compile and version.is_compiler_available():
                if isinstance(compile, dict):
                    compile_kwargs = compile
                if isinstance(compile, str):
                    compile_kwargs = dict(mode=compile)
                else:
                    compile_kwargs = {}
                compiled_module = torch.compile(wrapped_module, **compile_kwargs)

            # wrapped_modules.append(wrapped_module)
            compiled_modules.append(compiled_module)

        _return_as_tuple = len(compiled_modules) > 1
        if return_intermediates:
            return (
                tuple(original_modules) if _return_as_tuple else original_modules[0],
                tuple(parallel_modules) if _return_as_tuple else parallel_modules[0],
                tuple(compiled_modules) if _return_as_tuple else compiled_modules[0],
            )

        return tuple(compiled_modules) if _return_as_tuple else compiled_modules[0]

    def _wrap_module_legacy(
        self, module: nn.Module, mode: str, mixed_precision: bool | str | None, compile: str | bool | None
    ) -> nn.Module:
        wrap_kwargs = {}
        if self._state.is_distributed:
            if mode == 'ddp':
                factory = DistributedDataParallelFactory()
                wrap_kwargs['device_ids'] = [self.local_rank]
            elif mode == 'fsdp':
                factory = FullyShardedDataParallelFactory()
                wrap_kwargs['mixed_precision'] = mixed_precision
                if compile and version.is_compiler_available():
                    wrap_kwargs['use_orig_params'] = True
            elif mode == 'none':
                factory = NoParallelFactory()
            else:
                raise Exception(f'Unknown data parallelizm mode "{mode}"')

        else:
            factory = NoParallelFactory()

        wrapped_module = factory.wrap_module(module, **wrap_kwargs)
        self._factories.append(factory)

        return wrapped_module

    def _wrap_module(
        self, module: nn.Module, mode: str, mixed_precision: bool | str | None, compile: str | bool | None
    ) -> nn.Module:
        if self._state.is_distributed:
            return simple.wrap_module(module, mode, mixed_precision, [self.rank], compile)
        else:
            return module

    def prepare_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        worker_init_fn: Callable | None = None,
        generator: torch.Generator = None,
    ) -> DataLoader:
        """Prepare dataset, given dataloader parameters.

        If the distributed packae is initialized, `DistributedSampler` is used as the `sampler`.

        Args:
        ----
            dataset (Dataset): Dataset
            batch_size (int): batch size.
            shuffle (bool, optional): Default: True.
            drop_last (bool, optional): Default: True.
            num_workers (int, optional): Default: 0.
            pin_memory (bool, optional): Default: True.
            worker_init_fn (Callable, optional): Default: None.
            generator (torch.Generator, optional): Default: None.

        Returns:
        -------
            DataLoader: the created dataloader

        """
        if self._state.is_distributed:
            # when distributed training, always use DistributedSampler
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=shuffle, drop_last=drop_last
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn,
                generator=generator,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn,
                generator=generator,
            )
        return dataloader

    def prepare_for_checkpointing(self, *optimizers, offload_to_cpu: bool = True) -> tuple | tuple[tuple]:
        """Prepare models and optimizers for checkpointing.

        Create two object that has {state_dict,load_state_dict} function, that returns/loads the state_dict like
        modules that are not wrapped, for the wrapped module and corresponding optimizer.
        This function is for checkpointing models without changing codes between different data parallel methods.
        Tested to work well with storch.checkpoint.Checkpoint

        Args:
        ----
            *optimizers: The optimizers used from training. If multiple models are used, the optimizers must be passed
                in the same order. If a model is not intended to be trained, pass None.
            offload_to_cpu (bool, optional): When the model is FSDP, configures is the weights are offload to cpu before
                serialization. When `True`, it also enables `rank0_only`, which will result in returning empty dicts on
                processes that are `rank!=0`. Default: True.

        Returns:
        -------
            tuple|tuple[tuple]: module state_dict {get,set}ter and optimizer state_dict {get,set}ter

        """
        if version.is_dist_state_dict_available():
            raise Exception('Use `prepare_stateful` for torch>=2.2.0.')
        assert len(optimizers) == len(self._factories)
        ckpt_ifs = []
        kwargs = (
            {'offload_to_cpu': offload_to_cpu}
            if isinstance(self._factories[0], FullyShardedDataParallelFactory)
            else {}
        )
        for i, optimizer in enumerate(optimizers):
            ckpt_ifs.append(self._factories[i].create_checkpoint_interface(optimizer, **kwargs))
        return tuple(ckpt_ifs) if len(ckpt_ifs) > 1 else ckpt_ifs[0]

    def prepare_stateful(
        self,
        module: nn.Module,
        optimizers: torch.optim.Optimizer,
        full_state_dict: bool = True,
        cpu_offload: bool = True,
        strict: bool = True,
    ) -> tuple:
        """Prepare models to be serialized.

        This function is for `torch>=2.2.0`. Use `prepare_for_checkpointing` in previous versions.
        Note that the arguments are different from the original legacy function.
        The changes are:
        - Does not support multiple inputs.
        - Requires the module as an additional input.

        Create two object that has {state_dict,load_state_dict} function, that returns/loads the state_dict like
        modules that are not wrapped, for the wrapped module and corresponding optimizer.
        This function is for checkpointing models without changing codes between different data parallel methods.
        Tested to work well with storch.checkpoint.Checkpoint.

        Args:
        ----
            module (nn.Module): _description_
            optimizers (torch.optim.Optimizer): _description_
            full_state_dict (bool, optional): _description_. Defaults to True.
            cpu_offload (bool, optional): _description_. Defaults to True.
            strict (bool, optional): _description_. Defaults to True.

        Returns:
        -------
            tuple: _description_

        """
        if not version.is_dist_state_dict_available():
            raise Exception('Use `prepare_for_checkpointing` for torch<=2.1.x.')

        return simple.create_checkpoint_interface(
            model=module,
            optim=optimizers,
            strategy=self.mode,
            full_state_dict=full_state_dict,
            cpu_offload=cpu_offload,
            strict=strict,
        )
