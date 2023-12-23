"""wandb."""

from __future__ import annotations

import os

from omegaconf import DictConfig

from storch.hydra_utils import to_object

try:
    import wandb
except ImportError:
    wandb = None


ENV_WANDB_API_KEY = 'WANDB_API_KEY'


def is_wandb_available():
    """Check if wandb is available."""
    return wandb is not None


def init(
    project: str,
    name: str | None = None,
    config: DictConfig | dict | None = None,
    tags: tuple[str, ...] | list[str] | None = None,
    resume: str | bool | None = True,
    sync_tensorboard: bool = False,
    group: str | None = None,
    entity: str | None = None,
) -> None:
    """Wrap `wandb.init` function.

    This function requires the `WANDB_API_KEY` environment to be set.
    If not, wandb will not be initialized and returns `None`.
    The changes from the original `wandb.init` are:

    - Requires `project` argument.

    - `config` can be an `omegaconf.DictConfig` object. If so, this object will be will be converted to python
        dict before passed to `wandb.init`.

    - Always sets `anonymous='never'`. In addition, it searches for an environment variable `WANDB_API_KEY`,
        and requires it to be set.

    - `resume` defaults to True. `wandb.finish` should be called explicitly to avoid bugs.

    - Removes some of the arguments I think I will never use. Please check the official documentation for
        the arguments that are omitted.


    Args:
    ----
        project (str): Project name.
        name (str | None, optional): name of the run. Default: None.
        config (DictConfig | dict | None, optional): config for the run. Default: None.
        tags (tuple[str,...] | list[str] | None, optional): tags for the run. Default: None.
        resume (str | bool | None, optional): resume logging from last run. Default: None.
        sync_tensorboard (bool, optional): sync tensorboard. Default: False.
        group (str | None, optional): group. Default: None.
        entity (str | None, optional): entity (usename). Default: None.

    Returns:
    -------
        Run: a `wandb.Run` object.
    """
    run = None
    if is_wandb_available() and os.getenv(ENV_WANDB_API_KEY, '') != '':
        if isinstance(config, DictConfig):
            config = to_object(config)

        run = wandb.init(
            anonymous='never',
            project=project,
            name=name,
            config=config,
            tags=tags,
            resume=resume,
            sync_tensorboard=sync_tensorboard,
            group=group,
            entity=entity,
        )

    return run


def finish(quiet: bool | None = None) -> None:
    """Wrap `wandb.finish`.

    Args:
    ----
        quiet (bool | None): Do not print summary. Default: None.
    """
    wandb.finish(quiet=quiet)
