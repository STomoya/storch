
from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from storch.distributed.factory import (DistributedDataParallelFactory,
                                        FullyShardedDataParallelFactory,
                                        NoParallelFactory)
from storch.torchops import convert_outputs_to_fp32
from storch.utils import version


class DistributedHelper:
    """Helper class for distributed training.

    Args:
        world_size (int, optional): world size. ignored when {torchrun,CPU,single GPU}. Default: 1.
        rank (int, optional): rank. ignored when {torchrun,CPU,single GPU}. Default: 0.

    Examples:
        >>> # When running with torchrun launcher, nothing needs to be set.
        >>> disthelper = DistributedHelper(world_size, rank) # arguments are ignored.
        >>> # When lauching with torch.multiprocessing.spawn
        >>> os.environ['MASTER_ADDR'] = '127.0.0.1'
        >>> os.environ['MASTER_PORT'] = '29500'
        >>> #   in main function
        >>> disthelper = DistributedHelper(world_size=world_size, rank=rank)
    """
    def __init__(self,
        world_size: int=1, rank: int=0
    ) -> None:
        self.backend = None
        self.master_addr = None
        self.master_port = None
        self.world_size = 1
        self.rank = 0
        self.device = torch.device('cpu')

        if torch.cuda.is_available(): # CUDA
            if self.is_torchrun():
                # torchrun can handle either single GPU and multiple GPU environment.
                self.master_addr = os.environ['MASTER_ADDR']
                self.master_port = os.environ['MASTER_PORT']
                dist.init_process_group(backend='nccl')
            elif torch.cuda.device_count() > 1:
                # we don't need distributed setup on single GPU environment.
                # lets require MASTER_ADDR and MASTER_PORT exported before execution.
                _ma, _mp = os.environ.get('MASTER_ADDR', None), os.environ.get('MASTER_PORT', None)
                # NOTE: should we fall back to single process, and not raise an exception?
                assert _ma is not None and _mp is not None, 'ENV "MASTER_ADDR" and "MASTER_PPORT" must be set.'
                dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

            if self.is_initialized():
                self.backend = dist.get_backend()
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()

            self.device = torch.device('cuda', self.rank)
            torch.cuda.set_device(self.device)

        self._mode = None
        self._factories = []


    @staticmethod
    def is_available() -> bool:
        return dist.is_available()

    @staticmethod
    def is_initialized() -> bool:
        return dist.is_initialized()

    @staticmethod
    def is_torchrun() -> bool:
        return dist.is_torchelastic_launched()

    def is_primary(self) -> bool:
        return self.rank == 0

    # gather functions.
    # these functions work on both single/multi GPU without any specifications.

    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """gather all tensors, scattered to multiple devices,
        to a tensor concatenated on the 0th dimension.

        Args:
            tensor (torch.Tensor): the tensor to gather

        Returns:
            torch.Tensor: gathered tensor
        """
        if self.is_initialized():
            if tensor.ndim == 0:
                tensor = tensor.clone()[None]
            output_tensor = torch.cat([torch.empty_like(tensor) for _ in range(self.world_size)], dim=0)
            dist.all_gather_into_tensor(output_tensor, tensor)
        else:
            output_tensor = tensor
        return output_tensor

    def gather_any(self, val: Any) -> torch.Tensor:
        """gather that supports float and int input.

        Args:
            val (Any): value to gather.

        Returns:
            torch.Tensor: gathered values as an tensor.
        """
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device=self.device)
        val = self.gather_tensor(val)
        return val

    def reduce_tensor(self, tensor: torch.Tensor, op: ReduceOp=ReduceOp.SUM) -> torch.Tensor:
        """reduce a tensor, scattered on multiple GPUs, with a certain operator.

        Args:
            tensor (torch.Tensor): the tensor to reduce
            op (ReduceOp, optional): the operator. Default: ReduceOp.SUM.

        Returns:
            torch.Tensor: the reduced tensor
        """
        if self.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    def reduce_any(self, val: Any, op: ReduceOp=ReduceOp.SUM) -> torch.Tensor:
        """reduce that supports float and int inputs.

        Args:
            val (Any): value to reduce
            op (ReduceOp, optional): operator. Default: ReduceOp.SUM.

        Returns:
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
        return self.reduce_any(val)

    def reduce_mean(self, val: Any) -> torch.Tensor:
        return self.reduce_any(val, ReduceOp.AVG)

    def reduce_prod(self, val: Any) -> torch.Tensor:
        return self.reduce_any(val, ReduceOp.PRODUCT)

    def reduce_min(self, val: Any) -> torch.Tensor:
        return self.reduce_any(val, ReduceOp.MIN)

    def reduce_max(self, val: Any) -> torch.Tensor:
        return self.reduce_any(val, ReduceOp.MAX)

    @staticmethod
    def barrier():
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def wait_for_all_processes():
        """torch.distributed.barrier with a recognizable name.
        """
        if dist.is_initialized():
            dist.barrier()


    def get_parallel_mode(self, mode: str=None):
        if self._mode is None:
            if mode is None:
                mode = 'ddp'
                if self.is_primary():
                    print(f'Data parallelizm default to "ddp".')

            assert mode.lower() in ['ddp', 'fsdp']
            self._mode = mode

        return self._mode


    def prepare_module(self, *modules: nn.Module, mode: str=None, mixed_precision: bool=False, compile: bool|str|dict|None=None) -> tuple[nn.Module]|nn.Module:
        """prepare the input modules with mode "mode".

        - If the device is a cuda device the module is first set to the device, then wrapped.
        - "mixed_precision" is enabled by wrapping the forward function with torch.cuda.amp.autocast,
            and storch.torchops.convert_outputc_to_fp32. No need to use autocast contextmanager.

        Args:
            *modules (nn.Module): modules to wrap for distributed training.
            mode (str, optional): data parallel mode. one of {ddp,fsdp} for distributed settings,
                any other for single GPU or CPU. Default: None.
            mixed_precision (bool, optional): use mixed precision. Default: False.
            compile (bool | str | dict, optional): compile the model via torch.compile.
                - if True, uses torch.compile without any arguments.
                - if dict, uses torch.compile and use dict assuming containing arguments (e.g. "mode")
                - if str,  uses torch.compile and input str as "mode" argument.
                Default: None.

        Raises:
            Exception: unknown data parallel mode.

        Returns:
            tuple[nn.Module]|nn.Module: the wrapped modules in the same order as input.
        """
        mode = self.get_parallel_mode(mode)

        wrapped_modules = []
        for module in modules:
            # confirm send module to device before wrapping the model
            module.to(self.device)

            # wrap the model. see storch.distributed.factory.
            wrap_kwargs = {}
            if self.is_initialized():

                if mode == 'ddp':
                    factory = DistributedDataParallelFactory()
                    wrap_kwargs['device_ids'] = [self.rank]
                elif mode == 'fsdp':
                    factory = FullyShardedDataParallelFactory()
                    wrap_kwargs['mixed_precision'] = mixed_precision
                    if compile and version.is_compiler_available():
                        wrap_kwargs['use_orig_params'] = True
                else:
                    raise Exception(f'Unknown data parallelizm mode "{mode}"')

            else:
                factory = NoParallelFactory()

            wrapped_module = factory.wrap_module(module, **wrap_kwargs)

            # wrap forward with autocast if enabled.
            if mixed_precision and self.device.type == 'cuda':
                wrapped_module._original_forward = wrapped_module.forward
                dtype = torch.float16
                forward = torch.cuda.amp.autocast(dtype=dtype)(wrapped_module.forward)
                wrapped_module.forward = convert_outputs_to_fp32(forward)

            self._factories.append(factory)

            # compile model
            if compile and version.is_compiler_available():
                if isinstance(compile, dict):
                    compile_kwargs = compile
                if isinstance(compile, str):
                    compile_kwargs = dict(mode=compile)
                else:
                    compile_kwargs = {}
                wrapped_module = torch.compile(wrapped_module, **compile_kwargs)

            wrapped_modules.append(wrapped_module)

        return tuple(wrapped_modules) if len(wrapped_modules) > 1 else wrapped_modules[0]


    def prepare_dataset(self, dataset: Dataset, batch_size: int, shuffle: bool=True, drop_last: bool=True,
        num_workers: int=0, pin_memory: bool=True, worker_init_fn: Callable=None, generator: torch.Generator=None
    ) -> DataLoader:
        """prepare dataset, given dataloader parameters.

        Args:
            dataset (Dataset): Dataset
            batch_size (int):
            shuffle (bool, optional): Default: True.
            drop_last (bool, optional): Default: True.
            num_workers (int, optional): Default: 0.
            pin_memory (bool, optional): Default: True.
            worker_init_fn (Callable, optional): Default: None.
            generator (torch.Generator, optional): Default: None.

        Returns:
            DataLoader: the created dataloader
        """
        if self.is_initialized():
            # when distributed training, always use DistributedSampler
            sampler = DistributedSampler(dataset,
                num_replicas=self.world_size, rank=self.rank,
                shuffle=shuffle, drop_last=drop_last
            )
            dataloader = DataLoader(dataset,
                batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=pin_memory,
                worker_init_fn=worker_init_fn, generator=generator
            )
        else:
            dataloader = DataLoader(dataset,
                batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                num_workers=num_workers, pin_memory=pin_memory,
                worker_init_fn=worker_init_fn, generator=generator
            )
        return dataloader


    def prepare_for_checkpointing(self, *optimizers, offload_to_cpu: bool=True) -> tuple|tuple[tuple]:
        """create two object that has {state_dict,load_state_dict} function, that returns/loads the state_dict like
        modules that are not wrapped, for the wrapped module and corresponding optimizer.
        This function is for checkpointing models without changing codes between different data parallel methods.
        Tested to work well with storch.checkpoint.Checkpoint

        Args:
            *optimizers: The optimizers used from training. If multiple models are used, the optimizers must be passed in the same order.
                If a model is not intended to be trained, pass None.
            offload_to_cpu (bool, optional): When the model is FSDP, configures is the weights are offload to cpu before serialization.
                When `True`, it also enables `rank0_only`, which will result in returning empty dicts on processes that are `rank!=0`.
                Default: True.

        Returns:
            tuple|tuple[tuple]: module state_dict {get,set}ter and optimizer state_dict {get,set}ter
        """
        assert len(optimizers) == len(self._factories)
        ckpt_ifs = []
        kwargs = {'offload_to_cpu': offload_to_cpu} if isinstance(self._factories[0], FullyShardedDataParallelFactory) else {}
        for i, optimizer in enumerate(optimizers):
            ckpt_ifs.append(self._factories[i].create_checkpoint_interface(optimizer, **kwargs))
        return tuple(ckpt_ifs) if len(ckpt_ifs) > 1 else ckpt_ifs[0]
