from __future__ import annotations

import os
from typing import ClassVar

import torch
import torch.distributed as dist


class DistributedState:
    """Singleton class for keeping distributed state for each process.

    The user doesn't need to touch this class except for special cases.

    originally from: https://github.com/huggingface/accelerate/blob/b16916f44795bd960bc734992d2819a955064935/src/accelerate/state.py#L95-L662
    hacked by: Tomoya Sawada
    """

    _shared_state: ClassVar[dict] = {}

    def __init__(self, cpu: bool = False, **kwargs) -> None:
        self.__dict__ = self._shared_state

        if not self.initialized:
            self.device = None
            self.backend = 'none'

            if int(os.environ.get('LOCAL_RANK', -1)) >= 0 and not cpu and torch.cuda.is_available():
                self.backend = kwargs.pop('backend', 'nccl')
                dist.init_process_group(self.backend, **kwargs)
                self.num_processes = dist.get_world_size()
                self.process_index = dist.get_rank()
                self.local_process_index = int(os.environ.get('LOCAL_RANK'))
                self.device = torch.device('cuda', self.local_process_index)
                torch.cuda.set_device(self.device)
            else:
                self.num_processes = 1
                self.process_index = self.local_process_index = 0
                self.device = torch.device('cpu') if cpu else torch.device('cuda')

    def __repr__(self) -> str:
        return (
            f'Backend: {self.backend}\n'
            f'Num processes: {self.num_processes}\n'
            f'Process index: {self.local_process_index}\n'
            f'Device: {self.device}\n'
        )

    @property
    def initialized(self):
        return self._shared_state != {}

    @property
    def is_distributed(self) -> bool:
        return self.num_processes > 1

    @property
    def is_main_process(self) -> bool:
        return self.process_index == 0
