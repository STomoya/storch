
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
from typing import Any, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from storch.path import Path


class Logger(object):
    '''Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file.
    from: https://github.com/NVlabs/stylegan3/blob/583f2bdd139e014716fc279f23d362959bcc0f39/dnnlib/util.py#L56-L112
    '''

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def write(self, text: Union[str, bytes]) -> None:
        '''Write text to stdout (and a file) and optionally flush.'''
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        '''Flush written text to both stdout and a file, if open.'''
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        '''Flush, close possible files, and remove stdout/stderr mirroring.'''
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None



class Collector:
    '''Collect values by summing values until .update() is called, then reset.
    '''
    def __init__(self) -> None:
        self._deltas = dict()

    @torch.no_grad()
    def report(self, name: str, value: float|int|torch.Tensor):
        '''Report a value with an identical name to trace.

        Arguments:
            name: str
                Key for the value. This name will be used at .add_scalar() of SummaryWriter.
            value: float|int|torch.Tensor
                The value to collect.
        '''
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


    def report_by_dict(self, step: dict[str, Any]):
        '''report values using a dict object. See .report() for details.'''
        for name, value in step.items():
            self.report(name, value)


    def mean(self, name):
        '''Return the mean value of the collected value. If not exist or total is 0, returns inf.'''
        if name not in self._deltas or self._deltas[name][0] == 0:
            return float('inf')
        return self._deltas[name][1] / self._deltas[name][0]


    def update(self):
        '''Return mean of all collected values and reset.'''
        output = {}
        for name in self._deltas.keys():
            mean = self.mean(name)
            if mean != float('inf'):
                output[name] = self.mean(name)
            self._deltas[name] = [0, 0]
        return output


class Status:
    '''Class for logging training status.

    Arguments:
        max_iters: int
            Maximum iterations to train.
        log_file: str
            Path to file for output logging to.
        bar: bool (default: False)
            Enable tqdm progress bar.
        log_interval: int (default: 1)
            Interval for logging status.
        logger_name: str (default: 'logger')
            The name of the logger.
        steptime_num_accum: int (default: 300)
            Number of iterations to accumulate for calculating the rolling ETA.
        tb_folder: str|None (default: None)
            Folder to save the tensorboard event.
            If not given, the parent folder of 'log_file' will be used.
        delta_format: str (default: '{key}: {value: 10.5f}')
            The format used to print the collected values.
              - key: The name used to identify the value.
              - value: The value.
    '''
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
        self._std_logger = Logger(log_file)
        logging.basicConfig(
            format='%(asctime)s | %(name)s | %(filename)s | %(levelname)s | - %(message)s',
            level=logging.INFO, stream=self._std_logger)
        self._logger = logging.getLogger(logger_name)
        self._delta_format = delta_format

        self._collector = Collector()

        self._step_start = time.time()
        self._steptime_num_accum = steptime_num_accum
        self._steptimes = []

        tb_folder = log_file.resolve().dirname() if tb_folder is None else tb_folder
        self._tbwriter = SummaryWriter(tb_folder)

        atexit.register(self._shutdown_logger)

    @property
    def max_iters(self):
        return self._max_iters
    @property
    def batches_done(self):
        return self._batches_done
    @batches_done.setter
    def batches_done(self, value):
        self._batches_done = value

    def get_kbatches(self, format='{kbatches:.2f}k'):
        kbatches = self._batches_done / 1000
        return format.format(kbatches=kbatches)

    '''print functions'''

    def print(self, *args, **kwargs):
        if not self._bar.disable:
            tqdm.write(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def log(self, message, level='info'):
        getattr(self._logger, level)(message)


    '''Information loggers'''

    def log_command_line(self):
        '''log command line used to execute the python script.'''
        command_line = sys.argv
        command_line = pprint.pformat(command_line)
        self.log(f'Execution command\n{command_line}')

    def log_args(self, args: Namespace, parser: ArgumentParser=None, filename: str=None):
        '''log argparse.Namespace obj.'''
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

    def log_omegaconf(self, config: DictConfig):
        '''log omegaconf.DictConfig obj.'''
        yamlconfig = OmegaConf.to_yaml(config)
        self.log(f'Config:\n{yamlconfig}')

    def log_dataset(self, dataloader: DataLoader):
        '''log dataset.'''
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

    def log_optimizer(self, optimizer: Optimizer):
        '''log optimizer.'''
        self.log(f'Optimizer:\n{optimizer}')

    def log_env(self):
        '''log pytorch build enviornment.'''
        env = get_pretty_env_info()
        self.log(f'PyTorch environment:\n{env}')

    def log_model(self, model):
        '''log nn.Module obj.'''
        self.log(f'Architecture: {model.__class__.__name__}:\n{model}')

    def log_gpu_memory(self):
        '''log memory summary.'''
        if torch.cuda.is_available():
            self.log(f'\n{torch.cuda.memory_summary()}')
        else:
            self.log('No GPU available on your enviornment.')

    def log_nvidia_smi(self):
        '''log nvidia-smi output.'''
        if torch.cuda.is_available():
            nvidia_smi_output = subprocess.run(
                'nvidia-smi', shell=True,
                capture_output=True, universal_newlines=True)
            self.log(f'\n{nvidia_smi_output.stdout}')
        else:
            self.log('No GPU available on your enviornment.')

    def log_stuff(self, *to_log):
        '''log information in one function'''
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
        '''update status'''
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
            self.log_nvidia_smi()

        self._bar.update(1)

        self._step_start = time.time()

    def tb_add_scalars(self, **kwargs):
        for key, value in kwargs.items():
            self._tbwriter.add_scalar(key, value, self.batches_done)

    def tb_add_images(self, tag, image_tensor, normalize=True, value_range=(-1, 1), nrow=8, **mkgridkwargs):
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
        '''Update accumulation values without updating iteration counts.'''
        self._collector.report_by_dict(kwargs)
        self.tb_add_scalars(**kwargs)


    @contextmanager
    def stop_timer(self, verbose=False):
        '''context manager to stop the timer.

        Example usage:
            # The first batch ETA will add the time of validation for the previous epoch
            for _ in range(epochs):
                train_one_epoch()   # calls .update() each step.
                val_on_each_epoch() # doesn't call .update()

            # To avoid this you can:
            for _ in range(epochs):
                train_one_epoch()
                with status.stop_timer(): # this will stop the timer during with statement.
                    val_on_each_epoch()
        '''
        stop_start = time.time()
        yield
        duration = time.time() - stop_start
        if verbose:
            self.log(f'TIMER[STOPPED]: duration: {duration}')
        self._step_start += duration

    def is_end(self):
        '''have reached last batch?'''
        return self.batches_done >= self.max_iters

    def _shutdown_logger(self):
        self.log('LOGGER: shutting down logger...')
        handlers = self._logger.handlers
        for handler in handlers:
            self._logger.removeHandler(handler)
            handler.close()
        self._std_logger.close()

    def load_state_dict(self, state_dict: dict) -> None:
        '''fast forward'''
        # load
        self._collector = state_dict['collector']
        self.batches_done = state_dict['batches_done']
        self._steptimes = state_dict['steptimes']
        if self.batches_done > 0:
            # fastforward progress bar if present
            if self._bar:
                self._bar.update(self.batches_done)

    def state_dict(self) -> dict:
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

    Usage:
        from storch.status import Status, ThinStatus
        from storch.dist_helper import DistributedHelper
        disthelper = DistributedHelper(rank, world_size)
        Status = Status if disthelper.is_primary() else ThinSatus
        status = Status(max_iter, ...)

        # then use the rest should work like Status.

    Arguments:
        max_iters: int
            maximum iteration to train
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

    def print(self, *args, **kwargs):
        '''print function'''
        print(*args, **kwargs)

    def log(self, message: str, level: str='info'):
        self.print(message)

    def log_command_line(self):
        pass

    def log_args(self, args: Namespace, parser: ArgumentParser=None, filename: str=None):
        pass

    def log_omegaconf(self, config: DictConfig):
        pass

    def log_dataset(self, dataloader: DataLoader):
        pass

    def log_optimizer(self, optimizer: Optimizer):
        pass

    def log_env(self):
        pass

    def log_model(self, model):
        pass

    def log_gpu_memory(self):
        pass

    def log_nvidia_smi(self):
        pass

    def log_stuff(self, *args):
        pass

    def update(self, **kwargs):
        self._batches_done += 1

    def initialize_collector(self, *keys):
        pass

    def update_collector(self, **kwargs):
        pass

    def tb_add_scalars(self, **kwargs):
        pass

    @contextmanager
    def stop_timer(self, verbose=False):
        yield

    def is_end(self):
        '''have reached last batch?'''
        return self.batches_done >= self.max_iters

    def load_state_dict(self, state_dict: dict) -> None:
        '''fast forward'''
        # load
        self.batches_done = state_dict['batches_done']

    def state_dict(self) -> dict:
        raise NotImplementedError()

    def plot(self, filename='loss'):
        pass
