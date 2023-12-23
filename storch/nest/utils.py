"""NeST utils."""

from functools import wraps


def _noops(*args, **kwargs):
    return


def _assert_initialized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.initialized, (
            f'NeST is not initialized. To call {func.__name__} you must call `NeST.initialize_training` '
            'before calling this function.'
        )
        return func(self, *args, **kwargs)

    return wrapper
