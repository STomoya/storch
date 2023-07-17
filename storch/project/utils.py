from __future__ import annotations

import sys
from typing import Callable

from omegaconf import DictConfig, OmegaConf

from storch import get_now_string
from storch.distributed import DistributedHelper, is_primary
from storch.hydra_utils import get_hydra_config, save_hydra_config
from storch.path import Folder, Path


def _default_folder_from_config(config: DictConfig) -> Folder:
    root_folder = Path(config.run.folder)
    name = config.run.name
    tag = config.run.get('tag', None)

    if tag is not None:
        if tag == 'date':
            tag = get_now_string()
        id = '.'.join([name, tag])
    else:
        id = name
    folder = Folder(root_folder / id)

    return folder


def init_run(
    config_file: str, child_folders: dict={}, save_config: bool=True,
    disthelper: DistributedHelper=None,
    get_folder_from_config: Callable=_default_folder_from_config,
    config_dir: str='config', default_config_file: str='config.yaml',
) -> tuple[DictConfig, Folder]:
    """Load config, make workspace dir, and save config.
    Optionally resume using saved config file inside a workspace dir.

    Args:
        config_file (str): name of the config file tobe saved.
        child_folders (dict, optional): child folders inside root dir. Default: {}.
        save_config (bool): save config. pass false on child processes. Default: False
        get_folder_from_config (Callable, optional): function returning a Folder object.
            Default behavior requires "run", "run.folder", "run.name" and optionally "run.tag".
            Returns "{run.folder}/{run.name}(.{run.tag})"
            Default: _default_folder_from_config.
        config_dir (str, optional): root dir to config files. Default: 'config'.
        default_config_file (str, optional): base config filename. Default: 'config.yaml'.

    Returns:
        DictConfig: loaded config.
        Folder: Folder object.

    Examples:
        >>> # create new run.
        >>> $ python3 train.py
        >>> # resume from a checkpoint
        >>> $ python3 train.py ./path/to/checkpoint/config.yaml
    """
    cmdargs = sys.argv[1:]

    # for resuming:
    # $ python3 train.py　./path/to/config.yaml
    if len(cmdargs) == 1 and cmdargs[0].endswith(config_file):
        config_path = cmdargs[0]
        config = OmegaConf.load(config_path)
        folder = Folder(Path(config_path).dirname())
        if child_folders != {}:
            folder.add_children(**child_folders)

    # for a new run.
    else:
        config = get_hydra_config(config_dir, default_config_file)
        folder: Folder = get_folder_from_config(config)
        if child_folders != {}:
            folder.add_children(**child_folders)
        if is_primary():
            folder.mkdir()
            if save_config:
                save_hydra_config(config, folder.root / config_file)

    return config, folder
