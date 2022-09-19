
import os
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class DistributedHelper:
    """Helper class for distributed training.

    Args:
        rank (int): Rank of the process.
        world_size (int): World size.
        backend (str, optional): The backend to use. Default: 'nccl'.
        master_addr (str, optional): Master address. Default: '127.0.0.1'.
        master_port (str, optional): Master port. Default: '12355'.
    """
    def __init__(self,
        rank: int, world_size: int, backend: str='nccl',
        master_addr: str='127.0.0.1', master_port: str='12355'
    ) -> None:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    def cleanup(self) -> None:
        """Destroy process group. Call on exit of process.
        """
        dist.destroy_process_group()

    @staticmethod
    def is_available() -> bool:
        """Is distributed training available?
        """
        return dist.is_available()

    @staticmethod
    def is_initialized() -> bool:
        """Is distributed training initialized?
        """
        ''''''
        return dist.is_initialized()

    def is_primary(self) -> bool:
        """Is current process rank==0?
        """
        if self.is_available() and self.is_initialized():
            return self.rank == 0
        return True

    @property
    def rank(self) -> int:
        """Rank of current process
        """
        if self.is_available() and self.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def world_size(self) -> int:
        """World size
        """
        if self.is_available() and self.is_initialized():
            return dist.get_world_size()
        return 1

    def barrier(self) -> None:
        """Wrapped torch.distributed.barrier which returns instead of throwing an error if not initialized.
        """
        if not self.is_available() or not self.is_initialized():
            return
        dist.barrier()

    def prepare_ddp_model(self, model: nn.Module) -> DDP:
        """Wrap model with DDP

        Args:
            model (nn.Module): The module to wrap.

        Returns:
            DDP: Wrapped module.
        """
        return DDP(model, device_ids=[self.rank])

    def prepare_fsdp_model(self, model: nn.Module) -> None:
        """Wrap model with FSDP

        Args:
            model (nn.Module): The module to wrap.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError(f'FSDP is not currently supported. Wait until stable release.')

    def prepare_distributed_sampler(self, dataset: Dataset, shuffle: bool) -> DistributedSampler:
        """Create a DistributedSampler object based on a dataset.

        Args:
            dataset (Dataset): The dataset.
            shuffle (bool): whether to apply shuffling.

        Returns:
            DistributedSampler: A DistributedSampler object.
        """
        return DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)


def world_size_as_device_count() -> int:
    """Detect device count and use as world size.

    Returns:
        int: The device count detected by torch.
    """
    return torch.cuda.device_count()


def spawn(func: Callable, world_size: int, *args) -> None:
    """_summary_

    Args:
        func (Callable): Function to spawn. Requires the first argument to catch rank.
        world_size (int): World size.
        *args: Arguments passed to func.
    """
    if world_size == 1:
        func(*args)
    else:
        mp.spawn(
            func,
            args=args,
            nprocs=world_size
        )
