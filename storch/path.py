
from __future__ import annotations

from typing import Any, Callable
import glob
import os
import re
import datetime

import storch


_float_pattern = re.compile(f'[0-9]+.[0-9]*')
def find_last_float(path):
    return float(_float_pattern.findall(path)[-1])

_int_pattern   = re.compile(f'[0-9]')
def find_last_int(path):
    return int(_int_pattern.findall(path)[-1])


class Path(str):
    '''pathlib.Path like class but is a string object and with additional methods'''

    @property
    def stem(self) -> str:
        return os.path.splitext(self.name)[0]

    @property
    def suffix(self) -> str:
        return os.path.splitext(self.name)[-1]

    @property
    def name(self) -> str:
        return os.path.basename(self) if self.isfile() else ''

    def __truediv__(self, other: str) -> "Path":
        if not isinstance(other, str):
            raise TypeError(f'unsupported operand type(s) for /: "{self.__class__.__name__}" and "{other.__class__.__name__}"')
        return type(self)(os.path.join(self, other))

    def mkdir(self) -> None:
        '''make directory if self not exists'''
        if not os.path.exists(self):
            os.makedirs(self)

    def exists(self) -> bool:
        return os.path.exists(self)

    def expanduser(self) -> "Path":
        return type(self)(os.path.expanduser(self))

    def glob(self,
        recursive: bool=False, filter_fn: Callable|None=None,
        sort: bool=False, sortkey: Callable|None=None
    ) -> list[str]:
        '''calls glob.glob on self, and optionally sort the result

        Arguments:
            recursive: bool (default: False)
                If True, recursively glob inside subfolders
            filter_fn: Callable (default: None)
                Func to filter the globed result
            sort: bool (default: False)
                If True, sort the result
            sortkey: Callable (default: None)
                Func for key argument of sorted()
        '''
        glob_pattern = self / ('**/*' if recursive else '*')
        paths = glob.glob(glob_pattern, recursive=recursive)
        if isinstance(filter_fn, Callable):
            paths = [path for path in paths if filter_fn(path)]
        if sort:
            paths = sorted(paths, key=sortkey)
        return paths

    def isdir(self) -> bool:
        return os.path.isdir(self)

    def isfile(self) -> bool:
        return os.path.isfile(self)

    def resolve(self) -> "Path":
        return type(self)(os.path.realpath(os.path.abspath(self)))


class Folder(object):
    '''Class for easily handling paths inside a root directory.

    Usage:
        folder = Folder('./checkpoint')
        print(folder.root) # >>> './checkpoint'

        # add subfolders
        folder.add_children(image='subfolder1', model='subfolder2/subsubfolder')
        # subfolders can be accessed by name like attrs
        print(folder.image) # >>> './checkpoint/subfolder1'
        print(folder.model) # >>> './checkpoint/subfolder2/subsubfolder'

        # "/" operator can be used to join file/folder to root/sub folder
        print(folder.image / 'image.jpg') # >>> './checkpoint/subfolder1/image.jpg'

        # make all directories
        folder.make()

        # list all directories as dict
        folder.list()
    '''
    def __init__(self, root: str, identify: bool=False, id: str=None) -> None:
        if identify:
            id = id if id is not None else datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
            root += '_'+id

        self._roots = storch.EasyDict()
        self._roots.root = Path(root)

    def __getattr__(self, __name: str) -> Any:
        try:
            if __name in self._roots:
                return self._roots[__name]
            return self.__dict__[__name]
        except KeyError:
            raise AttributeError(__name)

    def add_children(self, **kwargs) -> None:
        for name, folder in kwargs.items():
            self._roots[name] = self._roots.root / folder

    def mkdir(self) -> None:
        for name in self._roots:
            self._roots[name].mkdir()

    def list(self) -> dict:
        return self._roots
