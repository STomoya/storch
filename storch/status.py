'''
Collect training status.
'''

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import logging
import time, datetime
from statistics import mean
import warnings
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.collect_env import get_pretty_env_info
from tqdm import tqdm

'''Value Collector'''

class Meter(list):
    '''collect values'''
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def x(self, total: int=None):
        '''return x axis for plot

        Arguments:
            total: int (default: None)
                total length of x axis
                if given, x will be evenly distributed.
        '''
        if total is None:
            return range(1, self.__len__()+1)
        per_element = total // self.__len__()
        return range(per_element, total+1, per_element)

class Group(dict):
    def max_length(self):
        return max([len(v) for v in self.values()])

class Collector:
    '''Collect scalar values and plot them

    Structure:
        {
            'group1': {
                'key1' : [...],
                'key2' : [...]},
            ...
        }
    same group => will be plotted in same graph.

    Usage:
        key1 = 'Loss/train/g'
        key2 = 'Loss/train/d'
        #      |----------|-|
        #         group   | key

        collector = Collector()
        # initialize collector
        collector.initialize(key1, key2)
        # add values
        collector['Loss/train/g'].append(random.random())
        collector['Loss/train/d'].append(random.random())
        # plot
        collector.plot()
        # => image of 1x<number of groups> graph
    '''
    def __init__(self) -> None:
        self._groups = {}
        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    def _split_key(self, key: str) -> tuple[str, str]:
        key = key.split('/')
        return '/'.join(key[:-1]), key[-1]

    def initialize(self, *keys) -> None:
        for key in keys:
            self[key] = Meter(key)
        self._initialized = True

    def update_by_dict(self, step: dict):
        for key, value in step.items():
            self[key].append(value)

    def plot(self, filename: str='graph.jpg') -> None:
        col = self.__len__()

        fig, axes = plt.subplots(1, col, figsize=(7*col, 5), tight_layout=True)

        for i, group_name in enumerate(self):
            if col == 1: ax = axes
            else:        ax = axes[i]

            group = self[group_name]
            length = group.max_length()
            legends = []
            for key in group:
                legends.append(key)
                x, y = group[key].x(length), group[key]
                ax.plot(x, y)

            ax.set_title(group_name)
            ax.legend(legends, loc='upper right')
            ax.set_xlabel('iterations')

        plt.savefig(filename)
        plt.close()

    '''magic funcs'''

    def __getitem__(self, key: str) -> Any:
        if key in self._groups:
            return self._groups[key]
        group, key = self._split_key(key)
        return self._groups[group][key]

    def __setitem__(self, key: str, value: Any) -> None:
        group, key = self._split_key(key)
        if group not in self._groups:
            self._groups[group] = Group()
        self._groups[group][key] = value

    def __iter__(self) -> Iterable:
        return self._groups.__iter__()

    def __len__(self) -> int:
        return self._groups.__len__()

    def __str__(self) -> str:
        return self._groups.__str__()

'''Training Status'''

class Status:
    '''Status
    A class for keeping training status

    Arguments:
        max_iters: int
            maximum iteration to train
        bar: bool (default: True)
            if True, show bar by tqdm
        log_file: str (default: None)
            path to the log file
            if given, log status to a file
        log_interval: int (default: 1)
            interval for writing to log file
        logger_name: str (default: 'logger')
            name for logger
    '''
    def __init__(self,
        max_iters: int, bar: bool=True,
        log_file: str=None, log_interval: int=1, logger_name: str='logger',
        steptime_num_accum: int=100,
        tensorboard: bool=False, tb_folder: str|None=None
    ) -> None:

        self._bar = tqdm(total=max_iters) if bar else None
        self._max_iters    = max_iters
        self._batches_done = 0
        self._collector    = Collector()
        self._log_file     = log_file

        # logger
        # Remove handlers.
        # NOTE: This does not support two or more Status object at the same time,
        #       but supports when Status objects exists at different time.
        _root_logger = logging.getLogger()
        for hdlr in _root_logger.handlers:
            _root_logger.removeHandler(hdlr)
        if log_file is not None:
            logging.basicConfig(filename=log_file, filemode='w',
                format='%(asctime)s:%(filename)s:%(levelname)s: %(message)s')
            self._logger = logging.getLogger(logger_name)
            self._logger.setLevel(logging.DEBUG)
        self._log_interval = log_interval

        # timer
        self._step_start = time.time()
        self._steptime_num_accum = steptime_num_accum
        self._steptimes = []

        # tensorboard
        self._tb_writer = SummaryWriter(tb_folder) if tensorboard else None


    @property
    def max_iters(self):
        return self._max_iters
    @property
    def batches_done(self):
        return self._batches_done
    @batches_done.setter
    def batches_done(self, value):
        self._batches_done = value


    '''print functions'''

    def print(self, *args, **kwargs):
        '''print function'''
        if self._bar:
            tqdm.write(*args, **kwargs)
        else:
            print(*args, **kwargs)
    def log(self, message, level='info'):
        if hasattr(self, '_logger'):
            getattr(self._logger, level)(message)
        else:
            warnings.warn('No Logger. Printing to stdout.')
            self.print(message)


    '''Information loggers'''

    def log_args(self, args: Namespace, parser: ArgumentParser=None, filename: str=None):
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

    def log_dataset(self, dataloader: DataLoader):
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

    def log_env(self):
        env = get_pretty_env_info()
        self.log(f'PyTorch environment:\n{env}')

    def log_model(self, model):
        self.log(f'Architecture: {model.__class__.__name__}:\n{model}')

    def log_gpu(self):
        if torch.cuda.is_available():
            self.log(f'\n{torch.cuda.memory_summary()}')
        else:
            self.log('No GPU available on your enviornment.')

    def log_stuff(self, *to_log):
        '''log information in one function'''
        self.log_env()
        for obj in to_log:
            if isinstance(obj, DataLoader):
                self.log_dataset(obj)
            elif isinstance(obj, torch.nn.Module):
                self.log_model(obj)
            elif isinstance(obj, Namespace):
                self.log_args(obj)


    '''information acculation funcs'''

    def update(self, **kwargs) -> None:
        '''update status'''
        if not self._collector.initialized:
            self.initialize_collector(*list(kwargs.keys()))

        self.update_collector(**kwargs)

        _print_rolling_eta = False
        if len(self._steptimes) == self._steptime_num_accum:
            self._steptimes = self._steptimes[1:]
            _print_rolling_eta = True
        self._steptimes.append(time.time() - self._step_start)

        # log
        if self._log_file is not None and self.batches_done % self._log_interval == 0:
            message_parts = [
                f'STEP: {self.batches_done} / {self.max_iters}',
                f'INFO: {kwargs}']
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
        if self.batches_done == 0:
            # print gpu on first step
            # for checking memory usage
            self.log_gpu()

        if self._bar:
            postfix = [f'{k} : {v:.5f}' for k, v in kwargs.items()]
            self.bar.set_postfix_str(' '.join(postfix))
            self.bar.update(1)

        self.tb_add_scalars(**kwargs)

        self.batches_done += 1
        self._step_start = time.time()

    def initialize_collector(self, *keys):
        if not self._collector.initialized:
            self._collector.initialize(*keys)

    def update_collector(self, **kwargs):
        self._collector.update_by_dict(kwargs)
        self.tb_add_scalars(**kwargs)

    def tb_add_scalars(self, **kwargs):
        if self._tb_writer:
            for key, value in kwargs.items():
                self._tb_writer.add_scalar(key, value, self.batches_done)


    @contextmanager
    def stop_timer(self):
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
        self._step_start += time.time() - stop_start

    def is_end(self):
        '''have reached last batch?'''
        return self.batches_done >= self.max_iters


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
        self._collector.plot(filename)
