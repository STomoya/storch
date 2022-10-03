
from collections.abc import Iterable
from itertools import repeat

__all__=[
    'to_1tuple',
    'to_2tuple',
    'to_3tuple',
    'to_4tuple',
]


def _ntuple(n):

    def parse(x):
        if isinstance(x, Iterable):
            if len(x) != n:
                raise UserWarning(f'Expected input to have {n} elements. Got {len(x)}')
            if isinstance(x, tuple):
                return x
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
