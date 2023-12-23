class Loss:
    """Base class for loss functions.

    Args:
    ----
        return_all (bool): Return all values used to calculate the loss. Default: False
    """

    def __init__(self, return_all: bool = False) -> None:
        self._return_all = return_all
