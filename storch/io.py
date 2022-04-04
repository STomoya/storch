
from __future__ import annotations

import yaml

try:
    import ujson as json
except ImportError:
    import json

def load_json(path: str):
    '''Load a JSON format file'''
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data


def load_jsonl(path: str):
    '''Load a JSONL format file'''
    with open(path, 'r') as fin:
        lines = fin.read().strip().split('\n')
    data = [json.loads(line) for line in lines]
    return data


def dump_json(obj: dict, path: str, **kwargs):
    '''Dump a dict in JSON format'''
    with open(path, 'w') as fout:
        json.dump(obj, fout, **kwargs)


def dump_jsonl(obj: list[dict], path: str):
    '''Dump a List of Dict in JSONL format'''
    lines = [json.dumps(line) for line in obj]
    with open(path, 'w') as fout:
        fout.write('\n'.join(lines))


def load_yaml(path: str):
    '''Load a YAML format file'''
    with open(path, 'r') as fin:
        data = yaml.safe_load(fin)
    return data


def dump_yaml(obj: dict, path: str, **kwargs):
    '''Dump a Dict in YAML format'''
    with open(path, 'w') as fout:
        yaml.dump(obj, fout, **kwargs)
