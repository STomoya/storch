"""tqdm."""

# ruff: noqa: D103

from tqdm import tqdm as origtqdm
from tqdm import trange as origtrange


def tqdm(*args, **kwargs):
    if 'bar_format' not in kwargs:
        kwargs['bar_format'] = '{l_bar}{bar:15}{r_bar}'
    return origtqdm(*args, **kwargs)


def trange(*args, **kwargs):
    if 'bar_format' not in kwargs:
        kwargs['bar_format'] = '{l_bar}{bar:15}{r_bar}'
    return origtrange(*args, **kwargs)
