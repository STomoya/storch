"""Pure python utilities."""

import math

from stutil import (
    EasyDict,
    call_func_by_name,
    check_folder,
    construct_class_by_name,
    dynamic_default,
    get_now_string,
    get_obj_by_name,
    glob_inside,
    import_all_modules,
    natural_sort,
    prod,
    recursive_apply,
    save_command_args,
    save_exec_status,
)
from stutil.timer import Timer

__all__ = [
    'EasyDict',
    'Timer',
    'calc_num_sampling',
    'call_func_by_name',
    'check_folder',
    'construct_class_by_name',
    'dynamic_default',
    'get_now_string',
    'get_obj_by_name',
    'glob_inside',
    'import_all_modules',
    'natural_sort',
    'prod',
    'recursive_apply',
    'save_command_args',
    'save_exec_status',
]


def calc_num_sampling(high_resl: int, low_resl: int) -> int:
    """Calculate number of sampling times when scale factor is 2.

    Args:
        high_resl (int): Higher resolution.
        low_resl (int): Lower resolution.

    Returns:
        int: Number of sampling times.

    """
    return int(math.log2(high_resl) - math.log2(low_resl))
