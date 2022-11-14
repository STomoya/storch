"""Classes for keeping the best state_dict for easy saving and loading.
"""

import copy

import torch.nn as nn


class Compose:
    """Compose multiple state_dict keepers.

    Args:
        **keepers: state_dict keepers

    Examples:
        >>> model = build_model(...)
        >>> best_models = Compose(dict(
        ...     val_loss=BestStateKeeper('min', model),
        ...     accuracy=BestStateKeeper('max', model)
        ... ))
        >>> output = model(input)
        >>> loss = criterion(output, target)
        >>> accuracy = metric(output, target)
        >>> best_models.update(val_loss=loss, accuracy=accracy)
        >>> # to load the best state_dict call load with the name of the metric.
        >>> best_models.load('val_loss')
    """

    def __init__(self, **keepers) -> None:
        assert all(isinstance(keeper, BestStateKeeper) for keeper in keepers.values())
        self.keepers = keepers


    def update(self, step: int=None, **values):
        """update the state_dict keepers.

        Args:
            step (int, optional): current training step. Default: None.
            **values: values for updating the keepers.
        """
        for name, value in values.items():
            self.keepers[name].update(value, step=step)


    def load(self, name: str):
        """load the state_dict specified by "name" to the model.

        Args:
            name (str): the name of the keeper.
        """
        self.keepers[name].load()


    def state_dict(self):
        """composed state_dict. a dictionary containing all the keepers' state_dict."""
        return {name: keeper.state_dict() for name, keeper in self.keepers.items()}


    def load_state_dict(self, state_dict: dict):
        """load the state_dict of all keepers."""
        for name, keeper_state_dict in state_dict.items():
            self.keepers[name].load_state_dict(keeper_state_dict)


class BestStateKeeper:
    """Class for keeping the best state_dict, based on a certain metric.

    Args:
        direction (str): 'minimize' or 'maximize'.
        model (nn.Module): the model to keep the state_dict.
        init_abs_value (float): the absolute value to initialize the value. Default: 1e10

    Examples:
        >>> model = build_model(...)
        >>> # 'min' and 'max' indicates "smaller the better" and vise versa.
        >>> val_loss_model = BestStateKeeper('min', model)
        >>> loss = criterion(model(input), target)
        >>> # this will update the best state_dict if the value is smaller than current best value.
        >>> val_loss_model.update(loss)
        >>> # to load the best state_dict to the model just call.
        >>> val_loss_model.load()
    """
    def __init__(self, direction: str, model: nn.Module, init_abs_value: float=1e10) -> None:
        assert direction in ['min', 'max', 'minimize', 'maximize']
        self.direction = 'min' if direction in ['min', 'minimize'] else 'max'

        # reference to the model.
        if isinstance(model, nn.DataParallel):
            self.model = model.module
        else:
            self.model = model

        self.best_state_dict = self._copy_state_dict()
        self.value = (1 if self.is_minimize() else -1) * init_abs_value
        self.step = None


    def is_minimize(self) -> bool:
        """is direction minimize?

        Returns:
            bool: True if minimize.
        """
        return self.direction == 'min'


    def is_maximize(self) -> bool:
        """is direction maximize?

        Returns:
            bool: True if maximize.
        """
        return self.direction == 'max'


    def _copy_state_dict(self):
        return copy.deepcopy(self.model.state_dict())


    def update(self, new_value: float, step: int=None) -> bool:
        """update the state_dict if new_value is better that current value.

        Args:
            new_value (float): the calculated score to compare.
            step (int, optional): current training step. Default: None.

        Returns:
            bool: True if updated, else False
        """
        if self.is_minimize():
            if new_value < self.value:
                self.value = new_value
                self.best_state_dict = self._copy_state_dict()
                self.step = step
                return True

        if self.is_maximize():
            if new_value > self.value:
                self.value = new_value
                self.best_state_dict = self._copy_state_dict()
                self.step = step
                return True

        return False

    def load(self) -> None:
        """load the best state_dict to the model.
        """
        self.model.load_state_dict(self.best_state_dict)


    def state_dict(self) -> dict:
        """returns this class' state.

        Returns:
            dict: state_dict of this class
        """
        return dict(
            direction = self.direction,
            value = self.value,
            state_dict = self.best_state_dict,
            step = self.step
        )


    def load_state_dict(self, state_dict: dict):
        """loads state from specified state_dict.

        Args:
            state_dict (dict): state_dict.
        """
        self.direction = state_dict.get('direction')
        self.value = state_dict.get('value')
        self.best_state_dict = state_dict.get('state_dict')
        self.step = state_dict.get('step')
