"""Training status logger."""

from __future__ import annotations

import atexit
import datetime
import logging
import pprint
import subprocess
import sys
import time
import warnings
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from statistics import mean
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from stutil.logger import get_logger
from torch.optim import Optimizer
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from storch._funtext import ASCII_LOGO
from storch.path import Path
from storch.profiler import get_tb_profile, record_function


class Collector:
    '''Collect values by summing values until .update() is called, then reset.
    '''
    def __init__(self) -> None:
        self._deltas = dict()

    @torch.no_grad()
    def report(self, name: str, value: float|int|torch.Tensor) -> None:
        """Report a value with an identical name to trace.

        Args:
            name (str): Key for the value. This name will be used at .add_scalar() of SummaryWriter.
            value (float | int | torch.Tensor): The value to collect.
        """
        if value is None:
            return

        if name not in self._deltas:
            self._deltas[name] = [0, 0]

        if torch.is_tensor(value):
            value = value.detach().cpu()
            num = value.numel()
            total = value.sum().item()
        else:
            num, total = 1, value

        self._deltas[name][0] += num
        self._deltas[name][1] += total


    def report_by_dict(self, step: dict[str, Any]) -> None:
        """report values using a dict object. See .report() for details.

        Args:
            step (dict[str, Any]): dict of values to report.
        """
        for name, value in step.items():
            self.report(name, value)


    def mean(self, name: str) -> float:
        """Return the mean value of the collected value. If not exist or total is 0, returns inf.

        Args:
            name (str): Key used to report values.

        Returns:
            float: mean of the collected values.
        """
        if name not in self._deltas or self._deltas[name][0] == 0:
            return float('inf')
        return self._deltas[name][1] / self._deltas[name][0]


    def update(self) -> dict[str, float]:
        """Return mean of all collected values and reset.

        Returns:
            dict[str, float]: dict of mean of all reported values.
        """
        output = {}
        for name in self._deltas.keys():
            mean = self.mean(name)
            if mean != float('inf'):
                output[name] = self.mean(name)
            self._deltas[name] = [0, 0]
        return output


class Status:
    """Class for logging training status.

    Args:
        max_iters (int): Maximum iterations to train.
        log_file (str): Path to file for output logging to.
        bar (bool, optional): Enable tqdm progress bar. Default: False.
        log_interval (int, optional): Interval for logging status. Default: 1.
        logger_name (str, optional): The name of the logger. Default: 'logger'.
        steptime_num_accum (int, optional): Number of iterations to accumulate for calculating the rolling ETA.
            Default: 300.
        tb_folder (str | None, optional): Folder to save the tensorboard event.
            If not given, the parent folder of 'log_file' will be used. Default: None.
        delta_format (_type_, optional): The format used to print the collected values.
            - key: The name used to identify the value.
            - value: The value.
            Default: '{key}: {value: 10.5f}'.

    Examples::
        >>> status = Status(1000, './checkpoint/log.log')
        >>> while not status.is_end():
        >>>     output = model(input)
        >>>     # update training state. also updates the training progress.
        >>>     status.update(**{
        >>>         'Loss/CE/train': torch.rand(1),
        >>>         'Metrics/accuracy/train': torch.rand(1),
        >>>         'Scores/output': output, # tensors with any shapes can be passed.
        >>>         'Progress/lr': lr # float/int are also supported.
        >>>     })
        >>>     # update without updating training progress.
        >>>     status.dry_update(**{
        >>>         'Loss/CE/val': torch.rand(1),
        >>>         'Metrics/accuracy/val': torch.rand(1),
        >>>     })
    """
    def __init__(self,
        max_iters: int, log_file: str,
        bar: bool=False, log_interval: int=1, logger_name: str='logger',
        steptime_num_accum: int=300, tb_folder: str|None=None,
        delta_format: str='{key}: {value: 10.5f}'
    ) -> None:

        self._bar = tqdm(total=max_iters, disable=not bar)
        self._max_iters = max_iters
        self._batches_done = 0
        self._log_file = log_file
        self._log_interval = log_interval


        log_file = Path(log_file)
        self._logger = get_logger(logger_name, filename=log_file, mode='a',
            format='%(asctime)s | %(name)s | %(filename)s | %(levelname)s | - %(message)s')
        self._delta_format = delta_format

        self._collector = Collector()

        self._step_start = time.time()
        self._steptime_num_accum = steptime_num_accum
        self._steptimes = []

        self._tb_folder = log_file.resolve().dirname() if tb_folder is None else tb_folder
        self._tbwriter = SummaryWriter(self._tb_folder)
        self._profiler = None

        atexit.register(self._shutdown_logger)

        self.log('\n'+ASCII_LOGO)

    @property
    def max_iters(self):
        return self._max_iters
    @property
    def batches_done(self):
        return self._batches_done
    @batches_done.setter
    def batches_done(self, value):
        self._batches_done = value

    def get_kbatches(self, format='{kbatches:.2f}k') -> str:
        """Returns a formated kilo batches.

        Args:
            format (str, optional): format of the string. Default: '{kbatches:.2f}k'.

        Returns:
            str: The formated kilo batches.
        """
        kbatches = self._batches_done / 1000
        return format.format(kbatches=kbatches)

    '''print functions'''

    def print(self, *args, **kwargs) -> None:
        """Print function. If tqdm progress bar is enabled, uses tqdm.write as function.
        """
        if not self._bar.disable:
            tqdm.write(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def log(self, message: str, level='info') -> None:
        """log a message

        Args:
            message (str): The message to log.
            level (str, optional): log level. Default: 'info'.
        """
        getattr(self._logger, level)(message)


    '''Information loggers'''

    def log_command_line(self) -> None:
        """log command line used to execute the python script.
        """
        command_line = sys.argv
        command_line = pprint.pformat(command_line)
        self.log(f'Execution command\n{command_line}')

    def log_args(self, args: Namespace, parser: ArgumentParser=None, filename: str=None) -> None:
        """log argparse.Namespace obj.

        Args:
            args (Namespace): The command line arguments parsed by argparse.
            parser (ArgumentParser, optional): Parser used to parse the command line arguments.
                Used to display the default values. Default: None.
            filename (str, optional): A filename. if given the arguments will be saved to this file.
                Default: None.
        """
        message = '------------------------- Options -----------------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            if parser is not None and v != parser.get_default(k):
                comment = f'[default: {parser.get_default(k)}]'
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '------------------------- End ---------------------------'
        self.log(f'Command line arguments\n{message}')

        if filename is not None:
            with open(filename, 'w') as fout:
                fout.write(message)
                fout.write('\n')

    def log_omegaconf(self, config: DictConfig) -> None:
        """log omegaconf.DictConfig obj.

        Args:
            config (DictConfig): The config to log.
        """
        yamlconfig = OmegaConf.to_yaml(config)
        self.log(f'Config:\n{yamlconfig}')

    def log_dataset(self, dataloader: DataLoader) -> None:
        """log DataLoader obj.

        Args:
            dataloader (DataLoader): The DataLoader object to log.
        """
        loader_kwargs = dict(
            TYPE           = dataloader.dataset.__class__.__name__,
            num_samples    = len(dataloader.dataset),
            num_iterations = len(dataloader),
            batch_size     = dataloader.batch_size,
            shuffle        = isinstance(dataloader.batch_sampler.sampler, RandomSampler),
            drop_last      = dataloader.drop_last,
            num_workers    = dataloader.num_workers,
            pin_memory     = dataloader.pin_memory)
        message = '------------------------- Dataset -----------------------\n'
        for k, v in sorted(loader_kwargs.items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '------------------------- End ---------------------------'
        self.log(f'Dataset\n{message}')

    def log_optimizer(self, optimizer: Optimizer) -> None:
        """log optimizer obj.

        Args:
            optimizer (Optimizer): The optimizer object to log.
        """
        self.log(f'Optimizer:\n{optimizer}')

    def log_env(self) -> None:
        """log pytorch build enviornment.
        """
        env = get_pretty_env_info()
        self.log(f'PyTorch environment:\n{env}')

    def log_model(self, model: torch.nn.Module) -> None:
        """log nn.Module obj.

        Args:
            model (torch.nn.Module): The module to log.
        """
        self.log(f'Architecture: {model.__class__.__name__}:\n{model}')

    def log_gpu_memory(self) -> None:
        """log memory summary.
        """
        if torch.cuda.is_available():
            self.log(f'\n{torch.cuda.memory_summary()}')
        else:
            self.log('No GPU available on your enviornment.')

    def log_nvidia_smi(self) -> None:
        """log nvidia-smi output.
        """
        if torch.cuda.is_available():
            nvidia_smi_output = subprocess.run(
                'nvidia-smi', shell=True,
                capture_output=True, universal_newlines=True)
            self.log(f'\n{nvidia_smi_output.stdout}')
        else:
            self.log('No GPU available on your enviornment.')

    def log_actual_batch_size(self,
        batch_size_per_proc: int, gradient_accumulation_steps: int, world_size: int
    ) -> None:
        """log actual batch size per optimization step

        Args:
            batch_size_per_proc (int): batch size per process.
            gradient_accumulation_steps (int): gradient accumulation steps.
            world_size (int): world size.
        """
        real_batch_size = batch_size_per_proc * gradient_accumulation_steps * world_size
        if real_batch_size == batch_size_per_proc:
            return

        self.log(('Batch size:\n'
            f'--------------------------------------------------\n'
            f'            Batch size per process : {batch_size_per_proc}\n'
            f'       Gradient accumulation steps : {gradient_accumulation_steps}\n'
            f'               Number of processes : {world_size}\n'
            f'  ----------------------------------------------\n'
            f'  Batch size per optimization step : {real_batch_size}\n'
            f'--------------------------------------------------'
        ))

    def log_stuff(self, *to_log) -> None:
        """log information in one function.
        """
        self.log_env()
        self.log_command_line()
        for obj in to_log:
            if isinstance(obj, DataLoader):
                self.log_dataset(obj)
            elif isinstance(obj, torch.nn.Module):
                self.log_model(obj)
            elif isinstance(obj, Optimizer):
                self.log_optimizer(obj)
            elif isinstance(obj, Namespace):
                self.log_args(obj)
            elif isinstance(obj, DictConfig):
                self.log_omegaconf(obj)


    '''information accumulation funcs'''

    def update(self, **kwargs) -> None:
        """update status.
        """
        self._collector.report_by_dict(kwargs)
        self.batches_done += 1

        _print_rolling_eta = False
        if len(self._steptimes) == self._steptime_num_accum:
            self._steptimes = self._steptimes[1:]
            _print_rolling_eta = True
        self._steptimes.append(time.time() - self._step_start)

        # log
        if (self._log_file is not None
            and (
            (self.batches_done == 1) or
            (self.batches_done % self._log_interval == 0) or
            (self.batches_done <= 100 and self.batches_done % 5 == 0))
        ):
            delta = self._collector.update()
            delta_str = []
            for key, value in delta.items():
                delta_str.append(self._delta_format.format(key=key, value=value))
            message_parts = [
                f'STEP: {self.batches_done} / {self.max_iters}',
                f'INFO: {", ".join(delta_str)}']
            # ETA
            # NOTE: this ETA is not exact.
            #       dealed by avging multiple steps. (see rolling eta)
            duration = self._steptimes[-1]
            eta_sec  = int((self.max_iters - self.batches_done) * duration)
            eta      = datetime.timedelta(seconds=eta_sec)
            message_parts.append(f'ETA(sec): {eta}')
            # peak memory
            if torch.cuda.is_available():
                peak_mem_byte = torch.cuda.max_memory_allocated()
                peak_mem_M    = peak_mem_byte / 1024 / 1024
                message_parts.append(f'peak_mem(M): {peak_mem_M:.1f}')
            # rolling eta for more stable ETA
            if _print_rolling_eta:
                rolling_duration = mean(self._steptimes)
                rolling_eta_sec  = int((self.max_iters - self.batches_done) * rolling_duration)
                rolling_eta      = datetime.timedelta(seconds=rolling_eta_sec)
                message_parts.append(f'rolling_ETA(sec): {rolling_eta}')
            self.log(' '.join(message_parts))
            self.tb_add_scalars(**delta)
        if self.batches_done == 10:
            # print gpu after some batches
            # for checking memory usage
            with record_function('nvidia-smi'):
                self.log_nvidia_smi()

        self._bar.update(1)

        if self._profiler is not None:
            self._profiler.step()

        self._step_start = time.time()

    def tb_add_scalars(self, **kwargs) -> None:
        """add scalars to tensorboard.
        """
        for key, value in kwargs.items():
            self._tbwriter.add_scalar(key, value, self.batches_done)

    def tb_add_images(self, tag: str, image_tensor: torch.Tensor, normalize=True, value_range=(-1, 1), nrow=8, **mkgridkwargs) -> None:
        """Add image to tensorboard.

        Args:
            tag (str): tag.
            image_tensor (torch.Tensor): tensor of images.
            normalize (bool, optional): argument for make_grid(). Default: True.
            value_range (tuple, optional): argument for make_grid(). Default: (-1, 1).
            nrow (int, optional): argument for make_grid(). Default: 8.
            **mkgridkwargs: other keyword arguments for make_grid().
        """
        images = make_grid(image_tensor, normalize=normalize, value_range=value_range, nrow=nrow, **mkgridkwargs)
        self._tbwriter.add_images(tag, images, self.batches_done)


    def initialize_collector(self, *keys):
        warnings.warn(
            'initialize_collector is no logger needed due to full migration to SummaryWriter. This function will be erased in the future version.',
            DeprecationWarning)

    def update_collector(self, **kwargs):
        warnings.warn(
            'update_collector is renamed to dry_update for disambiguation. This function will be erased in the future version. Please use dry_update()',
            DeprecationWarning)
        self.dry_update(**kwargs)

    def dry_update(self, **kwargs):
        """Update accumulation values without updating iteration counts.
        """
        self._collector.report_by_dict(kwargs)
        self.tb_add_scalars(**kwargs)


    @contextmanager
    def profile(self, enabled=True):
        """Context manager to profile a code block using pytorch profiler module.

        Args:
            enabled (bool, optional): Boolean to enable/disable profiling. Default: True.
        """
        if enabled:
            self._profiler = get_tb_profile(self._tb_folder)
            self._profiler.start()

        yield

        if enabled:
            self._profiler.stop()
            self._profiler = None


    @contextmanager
    def stop_timer(self, verbose=False) -> None:
        """context manager to stop the timer.

        Args:
            verbose (bool, optional): Boolean to enable logging. Default: False.

        Examples::
            >>> while not status.is_end():
            >>>     train_epoch(...)
            >>>     # this will stop the timer until exiting the with statement.
            >>>     with status.stop_timer():
            >>>         validate(...)
        """
        stop_start = time.time()
        yield
        duration = time.time() - stop_start
        if verbose:
            self.log(f'TIMER[STOPPED]: duration: {duration}')
        self._step_start += duration

    def is_end(self) -> None:
        """have reached last batch?
        """
        return self.batches_done >= self.max_iters

    def _shutdown_logger(self) -> None:
        """Safely shutdown the loggers. This function will be automatically be called using atexit.
        """
        self.log('LOGGER: shutting down logger...')
        handlers = self._logger.handlers
        for handler in handlers:
            self._logger.removeHandler(handler)
            handler.close()

    def load_state_dict(self, state_dict: dict) -> None:
        """fast forward training status by the given state_dict.

        Args:
            state_dict (dict): a dictionary made by status.state_dict().
        """
        # load
        self._collector = state_dict['collector']
        self.batches_done = state_dict['batches_done']
        self._steptimes = state_dict['steptimes']
        if self.batches_done > 0:
            # fastforward progress bar if present
            if self._bar:
                self._bar.update(self.batches_done)

    def state_dict(self) -> dict:
        """make a dictionary to save current training status.

        Returns:
            dict: Dict containing the states.
        """
        return dict(
            collector=self._collector,
            batches_done=self.batches_done,
            steptimes=self._steptimes)


    def plot(self, filename='loss'):
        warnings.warn(
            'plot is no logger supported due to full migration to SummaryWriter. This function will be erased in the future version.',
            DeprecationWarning)


class ThinStatus:
    '''Thin implementation of Status.
    We will always have to check if the rank is 0 when printing or logging.
    This class is for avoiding this tiresome coding.

    difference:
        - no logging.
        - no loss accumulation.

    Args:
        max_iters (int): maximum iteration to train

    Eaxmples::
        >>> from storch.status import Status, ThinStatus
        >>> from storch.dist_helper import DistributedHelper
        >>> disthelper = DistributedHelper(rank, world_size)
        >>> Status = Status if disthelper.is_primary() else ThinSatus
        >>> status = Status(max_iter, ...)
        >>> # then use the rest should work like Status.
    '''
    def __init__(self,
        max_iters: int, *_args,  **_kwargs
    ) -> None:
        self._max_iters = max_iters
        self._batches_done = 0

    @property
    def max_iters(self):
        return self._max_iters
    @property
    def batches_done(self):
        return self._batches_done
    @batches_done.setter
    def batches_done(self, value):
        self._batches_done = value

    def get_kbatches(self, format='{kbatches:.2f}k') -> str:
        """Returns a formated kilo batches.

        Args:
            format (str, optional): format of the string. Default: '{kbatches:.2f}k'.

        Returns:
            str: The formated kilo batches.
        """
        kbatches = self._batches_done / 1000
        return format.format(kbatches=kbatches)

    def print(self, *args, **kwargs) -> None:
        pass

    def log(self, message: str, level='info') -> None:
        pass

    def log_command_line(self) -> None:
        pass

    def log_args(self, args: Namespace, parser: ArgumentParser=None, filename: str=None) -> None:
        pass

    def log_omegaconf(self, config: DictConfig) -> None:
        pass

    def log_dataset(self, dataloader: DataLoader) -> None:
        pass

    def log_optimizer(self, optimizer: Optimizer) -> None:
        pass

    def log_env(self) -> None:
        pass

    def log_model(self, model: torch.nn.Module) -> None:
        pass

    def log_gpu_memory(self) -> None:
        pass

    def log_nvidia_smi(self) -> None:
        pass

    def log_stuff(self, *to_log) -> None:
        pass

    def update(self, **kwargs) -> None:
        """update status."""
        self._batches_done += 1

    def tb_add_scalars(self, **kwargs) -> None:
        pass

    def tb_add_images(self, tag: str, image_tensor: torch.Tensor, normalize=True, value_range=(-1, 1), nrow=8, **mkgridkwargs) -> None:
        pass

    def initialize_collector(self, *keys):
        pass

    def update_collector(self, **kwargs):
        pass

    def dry_update(self, **kwargs):
        pass

    @contextmanager
    def profile(self, enabled=True):
        yield


    @contextmanager
    def stop_timer(self, verbose=False) -> None:
        yield

    def is_end(self) -> None:
        return self.batches_done >= self.max_iters

    def _shutdown_logger(self) -> None:
        pass

    def load_state_dict(self, state_dict: dict) -> None:
        # load batches_done from the state_dict saved at the primary process.
        self.batches_done = state_dict['batches_done']

    def state_dict(self) -> dict:
        return {}

    def plot(self, filename='loss'):
        pass
