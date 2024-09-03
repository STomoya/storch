"""Init project."""

import argparse
import glob
import os

from omegaconf import OmegaConf

from storch.project._config import get_default_config


def get_args():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='.', help='The folder to be used for the project.')
    parser.add_argument('--empty-config', default=False, action='store_true', help='Do not use default config.')
    return parser.parse_args()


def run():
    """Initialize a folder.

    Folder tree:
        ./
            <root folder name>/
            config/
                config.yaml
            docker/
                torch/
                    Dockerfile
                    requirements.txt
            workspace.local/
            .dockerignore
            .env
            codecheck.py
            compose.yaml
            README.md

    Raises:
        Exception: The folder is not empty.

    Example:
        $ python -m storch.commands.init_project

    """
    args = get_args()

    if len(glob.glob(os.path.join(args.folder, '*'))) > 0:
        raise Exception(f'The folder "{args.folder}" is not empty.')

    folders = ['config', 'docker', 'docker/torch', 'workspace.local']
    files = [
        'config/config.yaml',
        'docker/torch/Dockerfile',
        'docker/torch/requirements.txt',
        '.dockerignore',
        'compose.yaml',
        'codecheck.py',
        '.env',
        'README.md',
    ]

    fullname = os.path.abspath(args.folder)
    folders.append(fullname.split('/')[-1])

    for folder in folders:
        os.mkdir(os.path.join(args.folder, folder))
    for file in files:
        if file.endswith('config.yaml') and not args.empty_config:
            with open(os.path.join(args.folder, file), 'w') as fp:
                fp.write(OmegaConf.to_yaml(get_default_config()))
        else:
            open(os.path.join(args.folder, file), 'w').close()
