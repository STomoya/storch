"""Classes for keeping the best state_dict for easy saving and loading."""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn

from storch.distributed import utils as distutils


class KeeperCompose:
    """Compose multiple state_dict keepers.

    Examples:
        ```
        model = build_model(...)
        best_models = Compose(dict(
            val_loss=BestStateKeeper('min', model),
            accuracy=BestStateKeeper('max', model)
        ))
        output = model(input)
        loss = criterion(output, target)
        accuracy = metric(output, target)
        best_models.update(val_loss=loss, accuracy=accracy)
        # to load the best state_dict call load with the name of the metric.
        best_models.load('val_loss')
        ```

    """

    def __init__(self, **keepers) -> None:
        """KeeperCompose.

        Args:
            **keepers: state_dict keepers

        """
        assert all(isinstance(keeper, BestStateKeeper) for keeper in keepers.values())
        self.keepers = keepers

    def update(self, step: int | None = None, **values):
        """Update the state_dict keepers.

        Args:
            step (int, optional): current training step. Default: None.
            **values: values for updating the keepers.

        """
        for name, value in values.items():
            self.keepers[name].update(value, step=step)

    def load(self, name: str):
        """Load the state_dict specified by "name" to the model.

        Args:
            name (str): the name of the keeper.

        """
        self.keepers[name].load()

    def state_dict(self):
        """Composed state_dict. a dictionary containing all the keepers' state_dict."""
        return {name: keeper.state_dict() for name, keeper in self.keepers.items()}

    def load_state_dict(self, state_dict: dict):
        """Load the state_dict of all keepers."""
        for name, keeper_state_dict in state_dict.items():
            self.keepers[name].load_state_dict(keeper_state_dict)


class BestStateKeeper:
    """Class for keeping the best state_dict, based on a certain metric.

    Examples:
        ```
        model = build_model(...)
        # 'min' and 'max' indicates "smaller the better" and vise versa.
        val_loss_model = BestStateKeeper('min', model)
        loss = criterion(model(input), target)
        # this will update the best state_dict if the value is smaller than current best value.
        val_loss_model.update(loss)
        # to load the best state_dict to the model just call.
        val_loss_model.load()
        ```

    """

    def __init__(
        self,
        name: str,
        direction: str,
        model: nn.Module,
        folder: str = '.',
        init_abs_value: float = 1e10,
        disthelper=None,
    ) -> None:
        """BestStateKeeper.

        Args:
            name (str): Name of the metric.
            direction (str): 'minimize' or 'maximize'.
            model (nn.Module): the model to keep the state_dict.
            folder (str): folder to save the states.
            init_abs_value (float): the absolute value to initialize the value. Default: 1e10
            disthelper (None): deprecated.

        """
        assert direction in ['min', 'max', 'minimize', 'maximize']
        self.name = name
        self.direction = 'min' if direction in ['min', 'minimize'] else 'max'

        # reference to the model.
        if isinstance(model, nn.DataParallel):
            self.model = model.module
        else:
            self.model = model

        os.makedirs(folder, exist_ok=True)
        self.filename = os.path.join(folder, f'{name}.torch')
        self.value = (1 if self.is_minimize() else -1) * init_abs_value
        self.step = None

        if disthelper is not None and distutils.is_primary():
            warnings.warn(
                (
                    'This class does not require DistributedHelper anymore, '
                    'and the argument will be erased in future versions.'
                ),
                FutureWarning,
                stacklevel=1,
            )

    def is_minimize(self) -> bool:
        """Is direction minimize.

        Returns:
            bool: True if minimize.

        """
        return self.direction == 'min'

    def is_maximize(self) -> bool:
        """Is direction maximize.

        Returns:
            bool: True if maximize.

        """
        return self.direction == 'max'

    def update(self, new_value: float, step: int | None = None) -> bool:
        """Update the state_dict if new_value is better that current value.

        Args:
            new_value (float): the calculated score to compare.
            step (int, optional): current training step. Default: None.

        Returns:
            bool: True if updated, else False

        """
        if (self.is_minimize() and new_value < self.value) or (self.is_maximize() and new_value > self.value):
            self.value = new_value
            self.step = step
            self.save(self.filename, model_state_only=False)
            return True

        return False

    def save(self, filename: str, model_state_only: bool = True):
        """Save state to file.

        Args:
            filename (str): The file name to save to.
            model_state_only (bool, optional): Save only the model state. Default: True.

        """
        state_dict = self.state_dict(model_state_only=model_state_only)
        if distutils.is_primary():
            torch.save(state_dict, filename)

    def load(self) -> None:
        """Load the best state_dict to the model."""
        distutils.wait_for_everyone()
        state_dict = torch.load(self.filename)
        if 'state_dict' in state_dict:
            state_dict = state_dict.get('state_dict')
        self.model.load_state_dict(state_dict)

    def state_dict(self, model_state_only: bool = False) -> dict:
        """Return this class' state.

        Returns:
            dict: state_dict of this class

        """
        model_state = self.model.state_dict()

        if model_state_only:
            return model_state

        return dict(
            direction=self.direction, value=self.value, step=self.step, filename=self.filename, state_dict=model_state
        )

    def load_state_dict(self, state_dict: dict):
        """Load state from specified state_dict.

        Args:
            state_dict (dict): state_dict.

        """
        self.direction = state_dict.get('direction')
        self.value = state_dict.get('value')
        self.step = state_dict.get('step')
        self.filename = state_dict.get('filename')
