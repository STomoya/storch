'''matplotlib.pyplot utils'''

from __future__ import annotations

import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

__all__=[
    'plt_subplots',
    'ax_setter'
]


@contextmanager
def plt_subplots(nrows: int=1, ncols: int=1, format_axes: bool=True, filename: str=None, **kwargs):
    """context manager which makes and yield subplots, then optionally save, finally close the figure.

    Args:
        nrows (int, optional): Number of subplot rows. Default: 1.
        ncols (int, optional): Number of subplot columns. Default: 1.
        format_axes (bool, optional): Format axes output of plt.subplots to [[ax00, ...], [ax10, ...], ...].
            Default: True.
        filename (str, optional): If given, saves the figure. Default: None.
        **kwargs: keyword arguments for plt.sobplots()

    Yields:
        Figure: matplotlib.figure.Figure object made by plt.subplot().
        Axes|list[Axes]|list[list[Axes]]: list or object of matplotlib.axes.Axes made by plt.subplot().
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    if format_axes:
        # format axes to [[ax00, ax01, ...], [ax10, ax11, ...], ...]
        if nrows == ncols == 1:        # ax00
            axes = [[axes]]
        elif ncols > 1 and nrows == 1: # [ax00, ax01, ...]
            axes = [axes]
        elif nrows > 1 and ncols == 1: # [ax00, ax10, ...]
            axes = [[ax] for ax in axes]

    yield fig, axes

    if filename is not None:
        plt.savefig(filename)

    plt.close(fig)


def ax_setter(ax: Axes, *,
    title: str=None, legend: bool=False, legend_loc: str='best', axis_off: bool=False,
    xlabel: str=None, ylabel: str=None, grid: bool=False, tick_top: bool=False, tick_right: bool=False,
    xticks: list[float]=None, xtick_labels: list[str]=None, yticks: list[float]=None, ytick_labels: list[str]=None,
    xlim: list[float]=None, ylim: list[float]=None, xbound: list[float]=None, ybound: list[float]=None,
    invert_xaxis: bool=False, invert_yaxis: bool=False, xscale: str=None, yscale: str=None
):
    """call setters of Axes object.

    This is an inplace operation.

    Args:
        ax (Axes): matplotlib.axes.Axes object.
        title (str, optional): title of the axes. Default: None.
        legend (bool, optional): visualize legend. Default: False.
        legend_loc (str, optional): location of the legend. Default: 'best'
        axis_off (bool, optional): disable all axis. If true, all arguments related to axis are ignored. Default: False.
        xlabel (str, optional): label of x axis. Default: None.
        ylabel (str, optional): label of y axis. Default: None.
        grid (bool, optional): visualize grid. Default: False.
        tick_top (bool, optional): position xtick to top. Default: False.
        tick_right (bool, optional): position ytick to right. Default: False.
        xticks (list[float], optional): ticks of x axis. Default: None.
        xtick_labels (list[str], optional): labels of ticks of x axis. Default: None.
        yticks (list[float], optional): ticks of y axis . Default: None.
        ytick_labels (list[str], optional): labels of ticks of y axis. Default: None.
        xlim (list[float], optional): lower and upper limit of x axis. Default: None.
        ylim (list[float], optional): lower and upper limit of y axis. Default: None.
        xbound (list[float], optional): lower and upper bounds of x axis. Default: None.
        ybound (list[float], optional): lower and upper bounds of y axis. Default: None.
        invert_xaxis (bool, optional): invert x axis. Default: False.
        invert_yaxis (bool, optional): invert y axis. Default: False.
        xscale (str, optional): scale of x axis. Default: None.
        yscale (str, optional): scale of y axis. Default: None.
    """

    if title is not None:
        ax.set_title(title)

    if legend:
        _, l = ax.get_legend_handles_labels()
        if len(l) > 0:
            ax.legend(loc=legend_loc)
        else:
            warnings.warn('No legends. Pass "label" argument to plotting functions for visualizing legends.')

    if axis_off:
        ax.set_axis_off()
    else: # these will be disabled of axis is off.
        # grid
        if grid:
            ax.grid()

        # labels
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # tick position
        if tick_top:
            ax.xaxis.tick_top()
        if tick_right:
            ax.yaxis.tick_right()

        # tick values and labels
        if xticks is not None:
            ax.set_xticks(xticks, xtick_labels)
        if yticks is not None:
            ax.set_yticks(yticks, ytick_labels)

    # bound/limit
    if xbound is not None:
        assert isinstance(xbound, (tuple, list)) and len(xbound) == 2
        ax.set_xbound(*xbound)
    if ybound is not None:
        assert isinstance(ybound, (tuple, list)) and len(ybound) == 2
        ax.set_ybound(*ybound)

    if xlim is not None:
        assert isinstance(xlim, (tuple, list)) and len(xlim) == 2
        ax.set_xlim(*xlim)
    if ylim is not None:
        assert isinstance(ylim, (tuple, list)) and len(ylim) == 2
        ax.set_ylim(*ylim)

    # invert order
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    # axis scale
    # {linear,log,symlog,logit}
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
