

import argparse
import glob
import os

from omegaconf import OmegaConf

from storch.project._config import get_default_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='.')
    return parser.parse_args()


def run():
    args = get_args()

    if len(glob.glob(os.path.join(args.folder, '*'))) > 0:
        raise Exception(f'The folder "{args.folder}" is not empty.')

    folders = [
        'config',
        'docker',
        'docker/torch',
        'workspace.local'
    ]
    files = [
        'config/config.yaml',
        'docker/torch/Dockerfile',
        'docker/torch/requirements.txt',
        '.dockerignore',
        'compose.yaml',
        'codecheck.py',
        '.env',
        'README.md'
    ]

    folders.append(args.folder.split('/')[-1])

    for folder in folders:
        os.mkdir(os.path.join(args.folder, folder))
    for file in files:
        if file.endswith('config.yaml'):
            with open(os.path.join(args.folder, file), 'w') as fp:
                fp.write(OmegaConf.to_yaml(get_default_config()))
        else:
            open(os.path.join(args.folder, file), 'w').close()
