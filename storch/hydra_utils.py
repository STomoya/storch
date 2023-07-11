'''configuration with hydra without @hydra.main()

NOTE: This implementation disables hydra's convenient functions like logging
      and autosaving config files. Use @hydra.main() instead if you want full
      functionality. This was needed to avoid conflicts with logging in storch.status.Status.
'''

from __future__ import annotations

import sys

from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf


def resolve(config: DictConfig|ListConfig) -> DictConfig|ListConfig:
    """resolve the config.

    Args:
        config (DictConfig | ListConfig): the config to resolve.

    Returns:
        DictConfig | ListConfig: resolved config.
    """
    OmegaConf.resolve(config)
    return config


def to_object(config: DictConfig|ListConfig):
    """convert omegaconf objects to python objects recursively. This function always resolves the config
    before converting to python objects.
    omegaconf objects are `issubclass(DictConfig, dict) == issubclass(ListConfig, list) == False`. There
    are some cases were this behavior causes unexpected errors (e.g., isinstance(config, dict) is false).


    Args:
        config (DictConfig | ListConfig): the config to convert.

    Returns:
        dict | list: the converted config.
    """
    py_obj = OmegaConf.to_object(config)
    return py_obj


def get_hydra_config(config_dir: str, config_name: str, overrides: list[str]=sys.argv[1:], resolve: bool=True) -> DictConfig:
    """gather config using hydra.

    Args:
        config_dir (str): Relative path to directory where config files are stored.
        config_name (str): Filename of the head config file.
        overrides (list[str], optional): Overrides. Usually from command line arguments. Default: sys.argv[1:].

    Returns:
        DictConfig: Loaded config.
    """
    with initialize_config_dir(config_dir=to_absolute_path(config_dir), version_base=None):
        cfg = compose(config_name, overrides=overrides)
    if resolve:
        OmegaConf.resolve(cfg)
    return cfg


def save_hydra_config(config: DictConfig, filename: str, resolve: bool=True) -> None:
    """save OmegaConf as yaml file

    Args:
        config (DictConfig): Config to save.
        filename (str): filename of the saved config.
    """
    if resolve:
        OmegaConf.resolve(config)
    with open(filename, 'w') as fout:
        fout.write(OmegaConf.to_yaml(config))
