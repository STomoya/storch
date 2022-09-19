
from __future__ import annotations

import yaml

try:
    import ujson as json
except ImportError:
    import json

from storch.imageops import cv2_load_image, pil_load_image, torch_load_image


def load_json(path: str) -> dict:
    """Load a JSON format file.

    Args:
        path (str): The JSON file to load.

    Returns:
        dict: The loaded JSON data.
    """
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL format file.

    Args:
        path (str): The JSONL file to load.

    Returns:
        list[dict]: List of JSON data.
    """
    with open(path, 'r') as fin:
        lines = fin.read().strip().split('\n')
    data = [json.loads(line) for line in lines]
    return data


def dump_json(obj: dict, path: str, **kwargs) -> None:
    """Dump a dict in JSON format

    Args:
        obj (dict): The dict object to dump
        path (str): The file to dump the object to.
    """
    with open(path, 'w') as fout:
        json.dump(obj, fout, **kwargs)


def dump_jsonl(obj: list[dict], path: str) -> None:
    """Dump a List of Dict in JSONL format

    Args:
        obj (list[dict]): The list of dict objects to dump
        path (str): The file to dump the object to.
    """
    lines = [json.dumps(line) for line in obj]
    with open(path, 'w') as fout:
        fout.write('\n'.join(lines))


def load_yaml(path: str) -> dict:
    """Load a YAML format file

    Args:
        path (str): The YAML file to load.

    Returns:
        dict: The loaded YAML data.
    """
    ''''''
    with open(path, 'r') as fin:
        data = yaml.safe_load(fin)
    return data


def dump_yaml(obj: dict, path: str, **kwargs) -> None:
    """Dump a dict in YAML format

    Args:
        obj (dict): The dict object to dump.
        path (str): The file to dump the object to.
    """
    with open(path, 'w') as fout:
        yaml.dump(obj, fout, **kwargs)
