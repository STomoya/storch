"""Image utils."""

import numpy as np


def random_box(size: tuple, min_size: float = 0, max_size: float = 1.0, margin: int = 0):
    """Sample a random rectangle.

    Args:
        size (tuple): Size of the region.
        min_size (float, optional): Minimum size of the box, relative to size. Default: 0.
        max_size (float, optional): Maximum size of the box, relative to size. Default: 1..
        margin (int, optional): Margin. Default: 0.

    Returns:
        _type_: _description_

    """
    width, height = size
    min_x = min_y = margin
    max_x, max_y = width - margin, height - margin
    min_x_size, min_y_size = int(width * min_size), int(height * min_size)
    max_x_size, max_y_size = int(width * max_size), int(height * max_size)
    while True:
        x = np.sort(np.random.randint(min_x, max_x, size=(2,)))
        y = np.sort(np.random.randint(min_y, max_y, size=(2,)))
        if min_x_size <= (x[1] - x[0]) <= max_x_size and min_y_size <= (y[1] - y[0]) <= max_y_size:
            break
    return tuple(map(int, (x[0], y[0], x[1], y[1])))
