"""Metrics for classification."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from storch.visualization import ax_setter, plt_subplots


def test_classification(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: list | None = None,
    digits: int = 6,
    return_dict: bool = True,
    filename: str | None = None,
    print_fn: Callable = print,
) -> None | dict:
    """Calculate and visualizing scores for classification (mostly using sklearn).

    Args:
    ----
        targets (np.ndarray): Ground truth.
        predictions (np.ndarray): Predictions.
        labels (list, optional): List containing names of the classes.. Defaults to None.
        digits (int, optional): digits argument for classification_report. Default: 6.
        return_dict (bool, optional): return classification report as dict? Default: True.
        filename (str, optional): If provided, visualize the confusion matrix via matplotlib. Defaults to None.
        print_fn (Callable, optional): A callable for printing the results. Defaults to print.

    Returns:
    -------
        dict: classification report as dict if return_dict is True.
    """
    if labels is not None:
        labels = np.array(labels)
        targets = labels[targets]
        predictions = labels[predictions]

    print_fn('TEST')
    # precision, recall, F1, accuracy
    print_fn(f'Classification report:\n{classification_report(targets, predictions, digits=digits)}')
    if return_dict:
        cls_report_dict = classification_report(targets, predictions, output_dict=True)
    else:
        cls_report_dict = None

    # confusion matrix
    confmat = confusion_matrix(targets, predictions, labels=labels, normalize='true')
    print_fn(f'Confusion matrix:\n{confmat}')

    if filename is not None:
        print_percentage = len(np.unique(targets)) <= 10  # noqa: PLR2004
        print_ticks = len(labels) <= 25 if labels is not None else False  # noqa: PLR2004
        with plt_subplots(filename=filename, format_axes=False, tight_layout=True) as (fig, ax):
            ax.matshow(confmat, cmap='Blues')
            if print_percentage:
                for (i, j), value in np.ndenumerate(confmat):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center')
            ticks = range(len(labels)) if print_ticks else []
            tick_labels = labels if print_ticks else []
            ax_setter(
                ax,
                xlabel='prediction',
                ylabel='ground truth',
                xticks=ticks,
                xtick_labels=tick_labels,
                yticks=ticks,
                ytick_labels=tick_labels,
                xtick_rotation=90,
            )
        print_fn(f'Saved confusion matrix to {filename}')

    return cls_report_dict
