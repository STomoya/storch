"""NeST."""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from stutil.path import Path
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import storch
from storch._optimizer_step import OptimizerStep
from storch.checkpoint import Checkpoint
from storch.distributed import DistributedHelper
from storch.distributed import utils as distutils
from storch.metrics import BestStateKeeper
from storch.nest import utils as nestutils
from storch.scheduler import build_scheduler
from storch.status import Status, ThinStatus
from storch.torchops import get_grad_scaler, get_optimizer_step


class NeST:
    """NeST."""

    def __init__(
        self,
        project_folder: Path,
        strategy: Literal['none', 'ddp', 'fsdp'] = 'ddp',
        mixed_precision: bool | Literal['fp16', 'float16', 'half', 'bf16', 'bfloat16'] = False,
        grad_accum_steps: int = 1,
        compile: bool | str | dict = False,
    ):
        """NeST.

        Args:
            project_folder (Path): the path to save logs and checkpoints for this project.
            strategy (str, optional): data parallelizm strategy. should be one of 'ddp' or 'fsdp'. Default: 'ddp'
            mixed_precision (bool, optional): Flag for enable/disable AMP. only supports pytorch native AMP
                implementation. Default: False.
            grad_accum_steps (int, optional): number of gradient accumulation steps. Default: 1.
            compile (bool, optional): compile the model using `torch.compile`. only available for pytorch>=2.0.0.
                Default: False.

        """
        self._disthelper = DistributedHelper()

        self._project_folder = Path(project_folder)

        self._strategy = self._disthelper.get_parallel_mode(strategy)
        self._mixed_precision = {
            False: False,
            True: 'fp16',
            'fp16': 'fp16',
            'float16': 'fp16',
            'half': 'fp16',
            'bfloat16': 'bfloat16',
            'bf16': 'bfloat16',
        }[mixed_precision]  # convert arg to consistant name.
        self._grad_accum_steps = grad_accum_steps
        self._compile = compile

        # grad scaler. only enabled when amp dtype is torch.float16
        self._grad_scaler = get_grad_scaler(enabled=self._mixed_precision == 'fp16', is_fsdp=self._strategy == 'fsdp')

        # objects
        self._status = None
        self._checkpoint = None

        self._step_fn: dict[int, OptimizerStep] = {}

        self._train_dataset = None
        self._train_loader = None
        self._set_epoch_fn = None

        self._org_models = []
        self._para_models = []
        self._final_models = []
        self._best_model_keepers = {}
        self._primary_best = None
        self._optimizers = []

        self._initialized = False

    def __repr__(self) -> str:
        """Repr."""
        string = (
            'NeST(\n'
            f'    project_folder: {self.project_folder}\n'
            f'    strategy: {self.strategy}\n'
            f'    mixed_precision: {self.mixed_precision}\n'
            f'    compile: {self.compile}\n'
            'distributed\n'
            f'    backend: {self.backend}\n'
            f'    world size: {self.world_size}\n'
            f'    rank: {self.rank}\n'
            f'    local rank: {self.local_rank}\n'
        )

        if self.initialized:
            string += (
                'training\n' f'    training iterations: {self.max_iters}\n' f'    batches done: {self.batches_done}\n'
            )

        return string + ')'

    """NeST properties"""

    @property
    def initialized(self) -> bool:
        """Is NeST object initialized."""
        return self._initialized

    @property
    def project_folder(self) -> Path:
        """Project folder."""
        return self._project_folder

    @property
    def strategy(self) -> str:
        """Data parallel strategy."""
        return self._strategy

    @property
    def mixed_precision(self) -> bool:
        """Mixed precision."""
        return self._mixed_precision

    @property
    def compile(self) -> bool | str | dict:
        """Compile."""
        return self._compile

    def get_model(self, index: int, where: str = 'parallel') -> nn.Module:
        """Get models.

        It supports returning intermediate step of the building function. 'original' returns the original model,
        'parallel' returns the model wrapped via parallel wrappers, and 'final' returns the final built model. If no
        parallel and no `torch.compile` is enabled, all variants return the same model.

        Args:
            index (int): returns the `index`-th created model.
            where (str, optional): Default: 'parallel'.

        Returns:
            nn.Module: the model.

        """
        if index >= len(self._org_models):
            return None
        if where == 'original':
            return self._org_models[index]
        elif where == 'parallel':
            return self._para_models[index]
        elif where == 'final':
            return self._final_models[index]

    def get_step_fn(self, optimizer: optim.Optimizer) -> OptimizerStep:
        """Return optimizer_step function, built for an specific optimizer.

        Args:
            optimizer (optim.Optimizer): optimizer.

        Returns:
            Callable: The optimizer_step function.

        """
        return self._step_fn.get(id(optimizer), None)

    # a

    """distributed properties"""

    @property
    def is_distributed(self) -> bool:
        """Is distributed training enabled."""
        return self._disthelper._state.is_distributed

    @property
    def backend(self) -> str:
        """Distributed backend."""
        return self._disthelper.backend

    @property
    def world_size(self) -> int:
        """World size."""
        return self._disthelper.world_size

    @property
    def rank(self) -> int:
        """Global rank."""
        return self._disthelper.rank

    @property
    def local_rank(self) -> int:
        """Local rank."""
        return self._disthelper.local_rank

    @property
    def device(self) -> torch.device:
        """Device of the process."""
        return self._disthelper.device

    def is_primary(self) -> bool:
        """Is primary process."""
        return self._disthelper.is_primary()

    def is_torchrun(self) -> bool:
        """Is launched by `torchrun` command."""
        return self._disthelper.is_torchrun()

    """training properties"""

    @property
    def status(self) -> Status:
        """`storch.status.Status` object. Returns `None` if NeST not initialized."""
        return self._status

    @property
    def checkpoint(self) -> Checkpoint:
        """`storch.checkpoint.Checkpoint` object. Returns `None` if NeST not initialized."""
        return self._checkpoint

    @property
    def best_model_keepers(self) -> BestStateKeeper:
        """Best model keeper."""
        return self._best_model_keepers

    @property
    def max_iters(self) -> int:
        """Maximum training iterations. Returns `None` if NeST not initialized."""
        return self._status.max_iters if self._status is not None else None

    @property
    def batches_done(self) -> int:
        """Number of batches done. Returns `None` if NeST not initialized."""
        return self._status.batches_done if self._status is not None else None

    @property
    def epoch_index(self) -> int:
        """Computer friendly epoch count. First epoch equals to 0. Returns `None` if NeST not initialized."""
        if self._status is None:
            return
        return int(self.status.batches_done / self._num_iters_per_epoch)

    @property
    def epoch(self) -> int:
        """Human friendly epoch count. First epoch equals to 1. Returns `None` if NeST not initialized."""
        if self._status is None:
            return
        return self.epoch_index + 1

    @property
    def gradscaler(self):
        """GradScaler."""
        return self._grad_scaler

    def is_end(self) -> bool:
        """Has reached maximum training iterations."""
        return self._status.is_end() if self._status is not None else None

    """methods"""

    def build_dataloader(
        self,
        is_train: bool,
        dataset: Dataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 1,
        pin_memory: bool = False,
        worker_init_fn: Callable | None = None,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        """Build DataLoader object given a user defined Dataset.

        If the dataset is the train split, pass `is_train=True`. Currently supports only one training data split. This
        function disables `shuffle`, `drop_last`, `worker_init_fn`, and `generator` when `is_train=False`, resulting in
        unexpected behavior if the {validation,test} split have randomness.

        This function automatically sets the sampler object to `DistributedSampler` if distributed training is used.

        This function is used to determine the actual iterations per epoch considering the gradient accumulation steps,
        and the DataLoader.sampler.set_epoch function needed for shuffling the dataset on distributed training.

        Args:
            is_train (bool): If True, the dataset is used to determine the actual training iterations considering
                the gradient accumulation steps.
            dataset (Dataset): The user defined dataset.
            batch_size (int): batch size. Note that this batch size is batch size per process, not the total batch size
                considering the gradient accumulation steps and number of processes.
            shuffle (bool, optional): shuffle the dataset every epoch. Automatically set to `False` when
                `is_true=False`. Default: False.
            drop_last (bool, optional): drop last batch so that the batch size throughout the training becomes even.
                Automatically set to `False` when `is_train=False`. Default: False.
            num_workers (int, optional): number of data loading workers. Note that this argument sets number of workers
                per process. The recommended number is `<total cpu cores>/<# of datasets>/<# of processes>`. Default: 1.
            pin_memory (bool, optional): memory pinning for faster data loading. Automatically disabled when
                no GPU is available. Default: False.
            worker_init_fn (Callable | None, optional): function to initialize the data loader workers. Default: None.
            generator (Generator | None, optional): RNG for random numbers. Default: None.

        Returns:
            DataLoader: The built data loader.

        """
        if not is_train:
            # If not train split, we do not need to `shuffle` and `drop_last` batch.
            # We also do not need `worker_init_fn` and `generator` which is only used when randomness is needed.
            shuffle = False
            drop_last = False
            worker_init_fn = None
            generator = None

        dataloader = self._disthelper.prepare_dataset(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

        if is_train:
            assert self._set_epoch_fn is None, 'Multiple train data splits are not supported.'

            self._train_dataset = dataset
            self._train_loader = dataloader

            if isinstance(dataloader.sampler, DistributedSampler):
                self._set_epoch_fn = dataloader.sampler.set_epoch
            else:
                self._set_epoch_fn = nestutils._noops

            self._num_iters_per_epoch = len(dataloader)
            self._actual_iters_per_epoch = math.ceil(self._num_iters_per_epoch / self._grad_accum_steps)

        return dataloader

    def build_model(self, builder: Callable | str, **builder_kwargs) -> nn.Module:
        """Build model given the builder and keyword arguments.

        The `builder` can be either a Callable or str object. If `builder` is a str object, the string is used to
        import the `class` or `function`, then the imported callable is called with the keyword arguments. The
        resulting object must be a `nn.Module` object.

        The model is automatically wrapped via data parallelizm method according to the `strategy` argument if
        distributed training is enabled. Then the forward function of the model is wrapped with
        `torch.cuda.amp.autocast`, if `mixed_precision` is enabled. Finally the model is compiled via `torch.compile`
        if `compile` is enabled. Note that `torch.compile` is only available for `PyTorch>=2.0.0`.

        Args:
            builder (Callable | str): The builder.
            **builder_kwargs: Keyword arguments used to build the object.

        Returns:
            nn.Module: The built model.

        """
        if isinstance(builder, str):
            model = storch.construct_class_by_name(class_name=builder, **builder_kwargs)
        elif callable(builder):
            model = builder(**builder_kwargs)

        assert isinstance(
            model, nn.Module
        ), f"The builder's resulting object must be an `nn.Module` object. got {type(model)}"

        org_model, para_model, compiled = self._disthelper.prepare_module(
            model,
            mode=self._strategy,
            mixed_precision=self._mixed_precision,
            compile=self._compile,
            return_intermediates=True,
        )

        self._org_models.append(org_model)
        self._para_models.append(para_model)
        self._final_models.append(compiled)

        return compiled

    def build_optimizer(self, builder: Callable | str, parameters: OrderedDict, **builder_kwargs) -> optim.Optimizer:
        """Build the optimizer using the given builder, model parameters, and keyword arguments.

        If `builder` is a str object, the string is used to import the `class` or `function`, then the imported callable
        is called with the keyword arguments. The resulting object must be a `optim.Optimizer` object.

        Args:
            builder (Callable | str): the builder.
            parameters (OrderedDict): parameters updated by the optimizer.
            **builder_kwargs: Keyword arguments used to build the object.

        Returns:
            optim.Optimizer: The built optimizer

        """
        if isinstance(builder, str):
            optimizer = storch.construct_class_by_name(parameters, class_name=builder, **builder_kwargs)
        elif callable(builder):
            optimizer = builder(parameters, **builder_kwargs)

        assert isinstance(
            optimizer, optim.Optimizer
        ), f"The builder's resulting object must be an `optim.Optimizer` object. got {type(optimizer)}"

        self._optimizers.append(optimizer)

        return optimizer

    def build_scheduler(self, optimizer: optim.Optimizer, step_on_epoch: bool = False, **builder_kwargs) -> LRScheduler:
        """Builder learning rate scheduler.

        See `storch.scheduler.build_scheduler` for more details about the scheduler options. The scheduler defaults to
        call `scheduler.step()` on every iteration. To switch this behavior, pass `step_on_epoch=True`.

        Args:
            optimizer (optim.Optimizer): Optimizer to adjust learning rate using the built scheduler.
            step_on_epoch (bool, optional): call scheduler.step on each epoch. Default: False.
            **builder_kwargs: Keyword arguments used to build the object.

        Returns:
            LRScheduler: learning rate scheduler.

        """
        if builder_kwargs.get('num_iter_per_step', None) is None or not step_on_epoch:
            builder_kwargs['num_iter_per_step'] = 1 if step_on_epoch else self._actual_iters_per_epoch
        scheduler = build_scheduler(optimizer, **builder_kwargs)
        return scheduler

    def initialize_training(
        self,
        max_training_iters: int,
        log_file: str = 'log.log',
        log_interval: int = 100,
        ckpt_keep_last: int = 1,
        to_log: list | None = None,
        logger_name: str = 'logger',
        log_gpu_memory_at: list[int] | int | None = None,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        wandb_tags: list | None = None,
        wandb_config: dict | None = None,
        steptime_num_accum: int = 300,
        delta_format='{key}: {value: 10.5f}',
        ckpt_file_format: str = 'checkpoint.{count}.torch',
    ) -> Status:
        """Initialize training.

        This function must be called after calling all building functions.

        The initialization is done in the order:

        1. Build an object for keeping track of the training progress.

        2. Build optimizer step functions for each optimizer.

        3. Log the models, optimizers, and dataset used when training.

        4. Log batch size if distributed or gradient steps is bigger than 1.

        5. Build an object for checkpointing.

        6. Load latest checkpoint if available.

        Numbers 4 and 6 might not happen depending on the preferences.

        The optimizer step function created at step 2. is built using `storch.torchops.get_optimizer_step`.
        The function is created for all optimizers built using `build_optimizer` method. We cannot track other
        optimizers, so if you have to create an optimizer without using the method, you must create an
        optimizer step function manually or script the optimization code yourself. For clarification,
        `OptimizerStep` calls `backward` on the loss and `step` on the optimizer, also optionally zeros the grad,
        scale loss for AMP, clip gradient norm, and automatically handles gradient accumulation.
        See `storch._optimizer_step.OptimizerStep` for more details about the optimizer step function.

        This function returns a `storch.status.Status` object, that can be used to track training progress.
        See `storch.status.Status` for more details. NeST wraps most of the core functionalities (e.g.,
        `update`, `log`), so receiving and using this return object is optional. You also can access the same
        object via `status` attribute after initialization.


        Args:
            max_training_iters (int): maximum training iterations. Note that it is not epochs.
            log_file (str, optional): Output filename for logging. Default: 'log.log'.
            log_interval (int, optional): interval for logging training state. Default: 100.
            ckpt_keep_last (int, optional): keep last n checkpoints. If `None` all saved files are kept. Deafult: 1.
            to_log (list, optional): a list of objects to log. Default: [].
            logger_name (str, optional): Name of the logger. Default: 'logger'.
            log_gpu_memory_at (list[int] | int, optional): Log GPU memory at specified iteration. Default: None.
            wandb_project (str, optional): wandb project name. If None, disables wandb logging. wandb requires
                `WANDB_API_KEY` evironment variable. Default: None.
            wandb_name (str, optional): wandb name of run. Default: None.
            wandb_tags (list, optional): list of tags: Default: None.
            wandb_config (dict|DictConfig, optional): config of the run. accepts omegaconf objects. Default: None.
            steptime_num_accum (int, optional): number for how many step time to accumulate for logging rolling ETA.
                Default: 300.
            delta_format (str, optional): format for logging values. Default: '{key}: {value: 10.5f}'.
            ckpt_file_format (srt, optional): filename format for the checkpoint. The format must contain '{count}'.
                Default: 'checkpoint.{count}.torch'.

        Returns:
            Status: training status keeper.

        """
        if to_log is None:
            to_log = []
        StatusCls = Status if self._disthelper.is_primary() else ThinStatus
        self._status = StatusCls(
            max_iters=max_training_iters,
            log_file=self._project_folder / log_file,
            log_interval=log_interval,
            logger_name=logger_name,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            wandb_tags=wandb_tags,
            wandb_config=wandb_config,
            steptime_num_accum=steptime_num_accum,
            delta_format=delta_format,
        )

        for i, (dmodel, optimizer) in enumerate(zip(self._para_models, self._optimizers, strict=False)):
            # if gpu memory logging is enabled instantiate hook and register.
            if self.is_primary() and log_gpu_memory_at is not None:
                if isinstance(log_gpu_memory_at, int):
                    log_gpu_memory_at = [log_gpu_memory_at]
                stage_postfix = f' No. {i}' if len(self._optimizers) > 1 else ''
                post_backward_hook = [
                    self._status.log_gpu_memory('backward()' + stage_postfix, log_gpu_memory_at, as_hook=True)
                ]
                post_step_hook = [
                    self._status.log_gpu_memory('optimizer.step()' + stage_postfix, log_gpu_memory_at, as_hook=True)
                ]
            else:
                post_backward_hook = []
                post_step_hook = []

            self._step_fn[id(optimizer)] = get_optimizer_step(
                gradient_accumulation_steps=self._grad_accum_steps,
                num_iters_per_epoch=self._num_iters_per_epoch,
                module=dmodel,
                post_backward_hooks=post_backward_hook,
                post_optim_step_hooks=post_step_hook,
            )

        if len(self._org_models) > len(self._optimizers):
            to_log += self._org_models[len(self._optimizers) :]
        for model, optimizer in zip(self._org_models, self._optimizers, strict=False):
            to_log += [model, optimizer]
        self._status.log_stuff(*to_log, self._train_loader)
        self._status.log_actual_batch_size(self._train_loader.batch_size, self._grad_accum_steps, self.world_size)

        self._checkpoint = Checkpoint(self._project_folder, keep_last=ckpt_keep_last, filename_format=ckpt_file_format)
        self._checkpoint.register(nest_grad_scaler=self._grad_scaler, nest_status=self._status)

        self._initialized = True

        return self._status

    def set_best_model_keeper(self, name: str, direction: str, model: nn.Module) -> BestStateKeeper:
        """Setups best model tracker.

        The best model tracker can easily keep the state_dict of the best performing checkpoint with one function call.
        See `update_best_model` for the main API. You can set multiple best model keepers. If so, you must use
        different names to avoid replacing the previously defined object. Also you must pass the `name` argument to
        both `{update/load}_best_model`.

        Args:
            name (str): name of the evaluation scores.
            direction: (str): either `minimize` or `maximize`.
            model (nn.Module): The model to keep the best states.

        """
        self._best_model_keepers[name] = BestStateKeeper(
            name=name, direction=direction, model=model, folder=self.project_folder
        )
        if self._primary_best is None:
            self._primary_best = name

        self.register(**{f'nest_best_model_keepers_{name}': self._best_model_keepers[name]})

    """serialization methods"""

    def prepare_for_checkpointing(self, *optimizers, offload_to_cpu: bool = True) -> tuple | tuple[tuple]:
        """Create interface for serialization.

        Prepare for serialization by creating an interface which properly returns `state_dict` and loads `state_dict`,
        according to the parallelizm strategy.To use the checkpointing functionality provided by NeST with distributed
        setting, you must call this function before calling `register` and pass returned objects instead of the
        model/optimizer. If the corresponding model is not meant to be trained and no optimizer is built, pass a
        `None`. If not using distributed training, calling this function is not necessary and returns the input as-is
        if called.

        Usage:
            >>> nest = NeST(...)
            >>> model = nest.build_model(...)
            >>> optimizer = nest.build_optimizer(...)
            >>> nest.initialize_training(...)
            >>> model_if, optim_if = nest.prepare_for_checkpointing(optimizer)
            >>> nest.register(model=model_if, optim=optim_if)

        Args:
            *optimizers: The optimizers. If multiple optimizers are used, they must to passed in the same order
                as the corresponding `build_model` function call. Pass `None` is the model is not meant to be trained.
            offload_to_cpu (bool, optional): Offload state dict to CPU. Only affects 'fsdp'. Default: True.

        Returns:
            (tuple | tuple[tuple]): Interface for getting setting state_dict.

        """
        return self._disthelper.prepare_for_checkpointing(*optimizers, offload_to_cpu=offload_to_cpu)

    def prepare_stateful(
        self,
        module: nn.Module,
        optimizers: optim.Optimizer,
        full_state_dict: bool = True,
        cpu_offload: bool = True,
        strict: bool = True,
    ) -> tuple:
        """Create interface for serialization for `torch>=2.2.0`.

        Prepare for serialization by creating an interface which properly returns `state_dict` and loads `state_dict`,
        according to the parallelizm strategy.To use the checkpointing functionality provided by NeST with distributed
        setting, you must call this function before calling `register` and pass returned objects instead of the
        model/optimizer. If the corresponding model is not meant to be trained and no optimizer is built, pass a
        `None`. If not using distributed training, calling this function is not necessary and returns the input as-is
        if called.

        Usage:
            >>> nest = NeST(...)
            >>> model = nest.build_model(...)
            >>> optimizer = nest.build_optimizer(...)
            >>> nest.initialize_training(...)
            >>> model_if, optim_if = nest.prepare_stateful(model, optimizer)
            >>> nest.register(model=model_if, optim=optim_if)

        Args:
            module (nn.Module): The module.
            optimizers (optim.Optimizer) : The optimizer. Pass `None` if the model is not meant to be trained.
            full_state_dict (bool, optional): If True, return unsharded tensors for FSDP models. Default: True.
            cpu_offload (bool, optional): Offload state dict to CPU. Only affects 'fsdp'. Default: True.
            strict (bool, optional): same as `strict` option in `module.load_state_dict()`. Default: True.

        Returns:
            (tuple | tuple[tuple]): Interface for getting setting state_dict.

        """
        return self._disthelper.prepare_stateful(
            module=module,
            optimizers=optimizers,
            full_state_dict=full_state_dict,
            cpu_offload=cpu_offload,
            strict=strict,
        )

    @nestutils._assert_initialized
    def register(self, **kwargs) -> None:
        """Register objects for checkpointing.

        The object must have `state_dict` and `load_state_dict` method. You do not need to pass the
        `storch.status.Status` object returned by `NeST.initialize_training` which is automatically included if not
        given.

        Args:
            **kwargs: key, value pairs of objects to be registered. values must have `state_dict` and
                `load_state_dict` method.

        """
        for key, value in kwargs.items():
            if value is self._status:
                kwargs.pop(key)
        self._checkpoint.register(**kwargs)

    @nestutils._assert_initialized
    def save(self, **constants) -> None:
        """Save the checkpoint.

        Args:
            **constants: objects to save but does not have `state_dict` and `load_state_dict` method. You must
                overwrite the values manually.

        """
        self._checkpoint.save(**constants)

    @nestutils._assert_initialized
    def load_latest(self, map_location=None) -> dict:
        """Load the latest checkpoint."""
        return self._checkpoint.load_latest(map_location=map_location)

    def update_best_model(self, value: float, name: str | None = None) -> None:
        """Update best model.

        You can use this function to update the best performing model. The input values are automatically compared with
        the previous best value corresponding to the `direction` argument. If you have set multiple best model keepers,
        you should explicitly provide the name of the metric.

        Args:
            value (float): The value used to update the best model.
            name (str, optional): The name of the model. If only one best model keeper is set, this argument is not
                needed. If you have set multiple best model keepers this argument is required. Default: None.

        Raises:
            Exception: no best model keeper is set.

        """
        if len(self._best_model_keepers) == 0:
            raise Exception('You have not set any best model keepers. Call `set_best_model_keeper` beforehand.')

        if name is None:
            assert (
                len(self._best_model_keepers) == 1
            ), '`name` argument is required if you have set multiple best model keepers.'
            best_keeper = self._best_model_keepers[self._primary_best]
        else:
            best_keeper = self.best_model_keepers[name]
        best_keeper.update(value)

    def load_best_model(self, name: str | None = None) -> None:
        """Load best model.

        You can use this function to load the current best performing parameters to the model. If you have set multiple
        best model keepers, you should explicitly provide the name of the metric. It is okay to pass `None`, and if so
        the first best model keeper is used to load parameters.

        Args:
            name (str, optional): The name of the model. If only one best model keeper is set, this argument is not
                needed. If you have set multiple best model keepers and `None` is passed, the first model keeper is
                used to load parameters. Default: None.

        Raises:
            Exception: no best model keeper is set.

        """
        if len(self._best_model_keepers) == 0:
            raise Exception('You have not set any best model keepers. Call `set_best_model_keeper` beforehand.')

        if name is None:
            best_keeper = self._best_model_keepers[self._primary_best]
        else:
            best_keeper = self.best_model_keepers[name]
        best_keeper.load()

    """distributed methods"""

    def set_epoch(self, epoch: int | None = None) -> None:
        """Wrapper of `torch.utils.data.DistributedSampler.set_epoch`.

        Args:
            epoch (int, optional): epoch argument passed to `set_epoch`. If `None`, current epoch is determined
                by `floor(_status.batches_done / _num_iters_per_epoch)`. Default: None.

        """  # noqa: D401
        epoch = self.epoch_index if epoch is None else epoch
        self._set_epoch_fn(epoch)

    def barrier(self) -> None:
        """`torch.distributed.barrier`."""
        self._disthelper.barrier()

    def wait_for_all_processes(self) -> None:
        """`torch.distributed.barrier` with a recognizable name."""
        self.barrier()

    def on_primary(self, func: Callable, *args: Any, **kwargs: dict[str, Any]) -> Any | None:
        """Execute function only on primary process.

        Args:
            func (Callable): the function to execute.
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            (Any | None): return value of the function or `None` if not on primary process.

        """
        if self.is_primary():
            return func(*args, **kwargs)

    def gather(
        self, obj: Any, dst: int | None = None, into_tensor: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor] | tuple[Any]:
        """Wrapper of `distributed.utils.gather`.

        ## Original docstring

        gather objects between devices.

        Can be a torch.Tensor or a picklable python object.

        By default tensors are gathered into a single tensor. To gather into a list of tensors,
        set `into_tensor=False`. Python objects are not affected by this argument and are always
        gathered into a list.

        By default the objects are gathered to all devices. You can specify the device to gather
        to by passing a valid process index to the `dst` argument (e.g., 0). If `dst` argument
        is specified, `None` will be returned to all other processes.

        If is not a distributed environment, this function will just return the input `obj`.

        Args:
            obj (Any): object to gather. Can be a Tensor or picklable python object.
            dst (int, optional): destination device. If not given gathers to all devices. Default: None.
            into_tensor (bool, optional): If True and obj is a Tensor gather into a Tensor instead of a list.
                Default: True.

        Returns:
            (torch.Tensor | tuple[torch.Tensor] | tuple[Any]): gathered object.

        """  # noqa: D401
        return distutils.gather(obj=obj, dst=dst, into_tensor=into_tensor)

    def reduce(
        self, tensor: torch.Tensor, dst: int | None = None, op: distutils.ReduceOp = distutils.ReduceOp.SUM
    ) -> torch.Tensor:
        """Wrapper of `distributed.utils.gather`.

        ## Original docstring

        reduce a tensor according to the given `ReduceOp` enum.

        In contrast to `gather`, this function does not support python objects. If reducing
        a python number, convert object to a Tensor beforehand.

        By default the objects are reduced to all devices. You can specify the device
        by passing a valid process index to the `dst` argument (e.g., 0). If `dst` argument
        is specified, `None` will be returned to all other processes.

        If is not a distributed environment, this function will just return the input `obj`.

        Args:
            tensor (torch.Tensor): Tensor to reduce.
            dst (int, optional): destination device. If not given reduced to all device. Default: None.
            op (ReduceOp, optional): reduce option. Default: ReduceOp.SUM.

        Returns:
            torch.Tensor: reduced tensor.

        """  # noqa: D401
        return distutils.reduce(tensor=tensor, dst=dst, op=op)

    """training methods"""

    @nestutils._assert_initialized
    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler | None = None,
        module: nn.Module = None,
        *,
        clip_grad_norm: bool = False,
        max_norm: float = 5.0,
        zero_grad: bool = True,
        update_scaler: bool = True,
    ):
        """Back propagation and optimization step.

        This method calls optimizer step function according to the optimizer. As mentioned in
        `NeST.initialize_training`, optimizers must be created via `NeST.build_optimizer` method. For the arguments see
        `storch._optimizer_step.OptimizerStep` for more details.

        Args:
            loss (torch.Tensor): loss. Must be a scalar tensor.
            optimizer (optim.Optimizer): the optimizer to call `step` on.
            scheduler: (LRScheduler, optional): learning rate scheduler. This function assumes `step` is called every
                iteration. If the `step` is meant to be called on each epoch, you have to manually call `step`.
                Default: None.
            module (nn.Module, optional): module. Used for gradient clipping, so if `clip_grad_norm=True` this argument
                is required. Default: None.
            clip_grad_norm (bool, optional): enable gradient clipping. Default: False.
            max_norm (float, optional): `max_norm` argument for `torch.nn.utils.clip_grad_norm_`. Default: 5.0.
            zero_grad (bool, optional): zero gradients. Note that `set_to_none` is always set to `True` which is the
                default behavior after pytorch>=2.0.0. Defaut: True.
            update_scaler (bool, optional): update gradient scaler used when AMP is enabled. ignored when AMP is not
                enabled. Default: True.

        """
        self.get_step_fn(optimizer)(
            loss,
            optimizer,
            self._grad_scaler,
            scheduler,
            module,
            zero_grad=zero_grad,
            set_to_none=True,
            update_scaler=update_scaler,
            clip_grad_norm=clip_grad_norm,
            max_norm=max_norm,
        )

    def sync_grad_accum_steps(self, base: torch.optim.Optimizer):
        """Synchronize gradient accumulation steps.

        If there are only one OptimizerStep, this function will exit without doing anything.

        Args:
            base (torch.optim.Optimizer): Synchronize based on this optimizer.

        """
        if len(self._step_fn) <= 1:
            return

        source = self.get_step_fn(base)
        for target in self._step_fn.values():
            if target is source:
                continue
            target.accumulation_count = source.accumulation_count
            target.total_step_count = source.total_step_count

    """logger methods"""

    @nestutils._assert_initialized
    def update(self, **kwargs) -> None:
        """Update training progress and log.

        This function must be called every iteration to update the training progress. If not, you will not be able to
        use some methods and properties (e.g., `is_end`, `batches_done`) that depends on variables updated by this
        function.

        Also, training state logging is triggered by calling this function. You can configure the logging using
        arguments of `NeST.initialize_training`.

        Call this method with key and value pairs of objects you want to track. The objects can be pytorch tensors of
        any shape, python ints or floats. If a pytorch tensor is not a scalar the mean of all elements in the tensor
        will be calculated. The values are automatically reported to a tensorboard event. You can use the keys to group
        the tracked values for better visualization.
        See the [pytorch tensorboard docs](https://pytorch.org/docs/stable/tensorboard.html).

        Calling `NeST.update` or `NeST.status.update` will always increment `batches_done`. To report values without
        incrementation, use `NeST.dry_update` or `NeST.status.dry_update` instead.

        Args:
            **kwargs: key, value pairs for objects to track.

        Example:
            ```
            nest = NeST(...)
            # build objects
            nest.initialize_training(
                # configure logging with this function.
            )
            while not nest.is_end():
                # forward, update parameters, calc metrics, etc.
                nest.update(**{'Loss': loss, 'Accuracy': accuracy})
            ```

        """
        self._status.update(**kwargs)

    @nestutils._assert_initialized
    def dry_update(self, **kwargs) -> None:
        """Log but do not update training progress.

        Args:
            **kwargs: key, value pairs for objects to track.

        """
        self._status.dry_update(**kwargs)

    @nestutils._assert_initialized
    def log(self, message: str, level: str = 'info') -> None:
        """Log message using python logger.

        Args:
            message (str): the message to log
            level (str, optional): logging level. Defaults to 'info'.

        """
        self._status.log(message, level)

    @nestutils._assert_initialized
    def finish_wandb(self, quiet: bool | None = None) -> None:
        """Finish wandb logging. It is recommended to call this function explicitly to avoid bugs for resume.

        Args:
            quiet (bool, optional): do not log run stats. Default: None.

        """
        self._status.finish_wandb(quiet=quiet)

    @contextmanager
    @nestutils._assert_initialized
    def profile(self, enabled: bool = True) -> None:
        """Context manager for profiling."""
        with self._status.profile(enabled=enabled):
            yield

    @contextmanager
    @nestutils._assert_initialized
    def stop_timer(self, verbose: bool = False) -> None:
        """Context manager for stopping the timer."""
        with self._status.stop_timer(verbose=verbose):
            yield

    """utilities"""

    def to_device(self, obj: torch.Tensor | nn.Module) -> torch.Tensor | nn.Module:
        """Send object to device.

        Args:
            obj (torch.Tensor | nn.Module): object to send.

        Returns:
            (torch.Tensor | nn.Module): the object on the device.

        """
        return obj.to(self.device)
