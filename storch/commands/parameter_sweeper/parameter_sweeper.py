"""Parameter sweeper."""

import argparse
import json
import os
from functools import partial
from typing import Any

import yaml

# consts
# ------------------------------

COMMAND_TEMPLATE = """
# main command
if $test ; then
    {test_command}
else
    {command}
fi
"""


# template format.
PARAMETER_SWEEPER_TEMPLATE = """#!/bin/bash

test={do_test}

{variables}

{loop}
"""


_translation_table = str.maketrans({'.': '_', '-': '_', '/': '_'})


# ------------------------------
# helpers


def _argument_key_to_object_name(key: str) -> str:
    """Convert argument keys to variable names.

    Args:
    ----
        key (str): argument key.

    Returns:
    -------
        str: name of object used in shell script.

    Examples:
    --------
      1. --foo     => foo
      2. -f        => f
      4. --foo_bar => foo_bar
      4. --foo-bar => foo_bar
      5. foo.bar   => foo_bar
      6. foo/bar   => foo_bar
    """
    if key.startswith('-'):
        key = key.lstrip('-')
    key = key.translate(_translation_table)
    return key


def _argument_key_to_list_name(key: str):
    """Add "_list" at the end of the object_name indicating a list object.

    Args:
    ----
        key (str): argument key.

    Returns:
    -------
        str: name of list used in shell script.
    """
    object_name = _argument_key_to_object_name(key)
    return object_name + '_list'


def _convert_value(value: Any, type: str) -> str:
    """Convert value to string.

    Args:
    ----
        value (Any): value.
        type (str): either hydra or argparse.

    Returns:
    -------
        str: value string

    Examples:
    --------
        - iterables:
            type=='hydra':   [1, 2, 3] => "[1,2,3]"
            type=='argparse: [1, 2, 3] => "1 2 3"
        - others:
            str(value)
    """
    if type == 'hydra':
        if isinstance(value, list):
            return f'[{",".join(map(str, value))}]'
    elif type == 'argparse':
        if isinstance(value, list):
            return ' '.join(value)
    else:
        raise Exception(f'_convert_value: unknown argumanet type {type}.')

    return str(value)


def _for_loop(list_name: str, object_name: str, inner: str) -> str:
    """Create a for loop shell script.

    Args:
    ----
        list_name (str): name of iterable.
        object_name (str): name of object used to catch items from the iterable.
        inner (str): script running inside the for loop.

    Returns:
    -------
        str: a single for loop script.
    """
    iterator = '"${' + list_name + '[@]}"'
    return f'for {object_name} in {iterator}\n' 'do\n' f'{inner}\n' 'done'


def create_echo(object_names: list) -> str:
    """Create echo command script."""
    message = '\\t'.join(f'${x}' for x in object_names)
    return f'echo -e "{message}"'


def create_command(command: str, arguments: dict, argument_type: str, include_test: bool = True) -> str:
    """Create command string.

    Args:
    ----
        command (str): base command.
        arguments (dict): dictionary containing arguments of "command".
        argument_type (str): argument type. either hydra or argparse.
        include_test (bool, optional): include test command. Default: True.

    Returns:
    -------
        str: command with arguments.
    """
    command = f'{command}'
    separator = '=' if argument_type == 'hydra' else ' '
    num_tabs = 2 if include_test else 1

    object_names = []
    for key in arguments:
        object_name = _argument_key_to_object_name(key)
        object_names.append(object_name)
        shell_object = '${' + object_name + '}'
        command += ' \\\n' + '    ' * num_tabs + f'{key}{separator}{shell_object}'

    if include_test:
        echo_command = create_echo(object_names)
        return COMMAND_TEMPLATE.format(test_command=echo_command, command=command)

    return command


def create_variables(arguments: dict, argument_type: str, excludes: set | None = None) -> str:
    """Create variable definition scripts from arguments.

    Args:
    ----
        arguments (dict): dictionary containing arguments.
        argument_type (str, optional): argument type. either hydra or argparse.
        excludes (set): list of argument keys to exclude.

    Returns:
    -------
        str: variable definition script
    """
    if excludes is None:
        excludes = set()
    shell_script_lists = []
    for key, value in arguments.items():
        if isinstance(value, list) and key not in excludes:
            list_name = _argument_key_to_list_name(key)
            shell_script_value = ' '.join(map(partial(_convert_value, type=argument_type), value))
            shell_script_lists.append(f'{list_name}=({shell_script_value})')
        else:
            object_name = _argument_key_to_object_name(key)
            shell_script_value = _convert_value(value, type=argument_type)
            shell_script_lists.append(f'{object_name}={shell_script_value}')
    return '\n'.join(shell_script_lists)


def create_for_loop(command: str, arguments: dict, excludes: set | None = None) -> str:
    """Create for loop for parameter sweep.

    Args:
    ----
        command (str): the command to execute.
        arguments (dict): dictionary containing arguments.
        excludes (set): list of argument keys to exclude.

    Returns:
    -------
        str: for loop.
    """
    if excludes is None:
        excludes = set()
    full_loop = command
    for key, value in arguments.items():
        if isinstance(value, list) and key not in excludes:
            list_name = _argument_key_to_list_name(key)
            object_name = _argument_key_to_object_name(key)
            full_loop = _for_loop(list_name, object_name, full_loop)
    return full_loop


def generate_parameter_sweeper_script(
    main_command: str,
    arguments: dict,
    argument_type: str = 'hydra',
    excludes: list | None = None,
    include_test: bool = True,
) -> str:
    """Generate parameter sweeper script.

    Args:
    ----
        main_command (str): base command to run.
        arguments (dict): dictionary containing arguments.
        argument_type (str, optional): argument type. either hydra or argparse. Default: hydra.
        excludes (list, optional): list of argument keys to exclude. Default: [].
        include_test (bool, optional): include test command. Default: True.

    Returns:
    -------
        str: sweeper script.
    """
    if excludes is None:
        excludes = []
    shell_scripts = dict(variables=None, loop=None, do_test='true' if include_test else 'N/A')
    excludes = set(excludes)
    shell_scripts['variables'] = create_variables(arguments, argument_type, excludes)
    command = create_command(main_command, arguments, argument_type, include_test)
    shell_scripts['loop'] = create_for_loop(command, arguments, excludes)
    return PARAMETER_SWEEPER_TEMPLATE.format(**shell_scripts)


def _assert_config(config):
    """Check assertions."""
    assert 'main_command' in config, 'Missing required field "main_command".'
    assert 'arguments' in config, 'Missing required field "arguments".'
    assert isinstance(config['arguments'], dict), '"arguments" must be a dictionary'


def generate_from_json(path: str, include_test: bool = True) -> str:
    """Generate script from json file.

    Args:
    ----
        path (str): path to the config file.
        include_test (bool, optional): include test command. Default: True.

    Returns:
    -------
        str: parameter sweeper script.
    """
    with open(path, 'r') as fp:
        config = json.load(fp)
    _assert_config(config)
    return generate_parameter_sweeper_script(**config, include_test=include_test)


def generate_from_yaml(path: str, include_test: bool = True) -> str:
    """Generate script from yaml file.

    Args:
    ----
        path (str): path to the config file.
        include_test (bool, optional): include test command. Default: True.

    Returns:
    -------
        str: parameter sweeper script.
    """
    with open(path, 'r') as fp:
        config = yaml.safe_load(fp)
    _assert_config(config)
    return generate_parameter_sweeper_script(**config, include_test=include_test)


def get_args() -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--output', '-o', default='./sweep-parameters.sh', type=str)
    parser.add_argument('--no-test', default=False, action='store_true')
    args = parser.parse_args()
    return args


def main():  # noqa: D103
    args = get_args()
    config_file: str = args.config

    assert os.path.exists(config_file), f'"{config_file}" does not exist.'
    assert config_file.lower().endswith(('.yaml', '.yml', '.json')), 'Unsupported file format.'

    if config_file.lower().endswith(('.yaml', '.yml')):
        script = generate_from_yaml(config_file, not args.no_test)
    elif config_file.lower().endswith('.json'):
        script = generate_from_json(config_file, not args.no_test)

    with open(args.output, 'w') as fp:
        fp.write(script)
