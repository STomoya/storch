
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def test_classification(
    targets: np.ndarray, predictions: np.ndarray, labels: list=None,
    filename: str=None, print_fn: Callable=print
):
    '''Calculate and visualizing scores for classification (mostly using sklearn).

    Arguments:
        target: np.ndarray
            Ground truth.
        predictions: np.ndarray
            Predictions
        labels: list (default: None)
            List containing names of the classes.
        filename: str (default: None)
            If provided, visualize the confusion matrix via matplotlib.
        print_fn: Callable (default: print)
            A callable for printing the results.
    '''
    if labels is not None:
        labels = np.array(labels)
        targets = labels[targets]
        predictions = labels[predictions]

    print_fn('TEST')
    # precision, recall, F1, accuracy
    print_fn(
        f'Classification report:\n{classification_report(targets, predictions)}')

    # confusion matrix
    confmat = confusion_matrix(targets, predictions, labels=labels, normalize='true')
    print_fn(f'Confusion matrix:\n{confmat}')

    if filename is not None:
        # visualize confmat
        fig, ax = plt.subplots(tight_layout=True)
        ax.matshow(confmat, cmap='Blues')
        for (i, j), value in np.ndenumerate(confmat):
            ax.text(j, i, f'{value:.3f}', ha='center', va='center')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('prediction')
        ax.set_ylabel('ground truth')
        plt.savefig(filename)
        plt.close()
        print_fn(f'Saved confusion matrix to {filename}')
