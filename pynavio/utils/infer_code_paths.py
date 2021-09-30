import inspect
import re
import sys
import types
from pathlib import Path
from typing import List

from pigar.parser import parse_imports

from pynavio.utils.common import get_module_path
from pynavio.utils.directory_utils import _generate_default_to_ignore_dirs

RE_FIRST_MODULE_NAME = re.compile('(\.*[^.]+).*')


def _get_first_module_name(module_name: str):
    match = re.match(RE_FIRST_MODULE_NAME, module_name)
    if match:
        first_module_name = match.groups()[0]
    else:
        first_module_name = module_name
    return first_module_name


def _get_code_path(module_name: str, path: str):
    first_module_name = _get_first_module_name(module_name)
    path_parts = Path(path).parts
    # TODO: improve getting the path or document that will return the last occurrence of module name in the path
    index_of_folder_name_in_the_path = next(
        i for i in reversed(range(len(path_parts)))
        if path_parts[i] == first_module_name)
    return f'{Path( *path_parts[:index_of_folder_name_in_the_path+1])}'


def get_name_to_module_path_map(imported_modules, root_path, to_ignore_paths):
    name_to_module_path = dict()
    for module in imported_modules:
        name = module.name
        moduleobj = sys.modules.get(name, None)

        if inspect.ismodule(moduleobj) and getattr(moduleobj, '__file__',
                                                   None):
            if Path(root_path) in Path(get_module_path(
                    moduleobj)).parents and _is_not_in_ignore_paths(
                        moduleobj, to_ignore_paths):
                name_to_module_path[module.name] = get_module_path(moduleobj)
    return name_to_module_path


def _is_not_in_ignore_paths(module, to_ignore_paths):
    return not any([
        Path(to_ignore_path) in Path(get_module_path(module)).parents
        for to_ignore_path in to_ignore_paths
    ])


def infer_imported_code_path(path,
                             root_path: str,
                             to_ignore_paths: List[str] = None):
    # TODO: add docstring about limitation that does not include relative paths
    # Can result in duplicated copies in code_paths if the the imports are inconsistent,
    # e.g. in one place from pynavio.utils.common import get_module_path and in other place
    # from utils.common import get_module_path (with adding more paths to PYTHONPATH)

    if to_ignore_paths is None:
        to_ignore_paths = []

    for p in [path, root_path, *to_ignore_paths]:
        assert Path(p).exists(), f"{p} does not exist"

    if not to_ignore_paths:
        to_ignore_paths = _generate_default_to_ignore_dirs(root_path)

    imported_modules, _ = parse_imports(
        path,
        ignores=[f'{to_ignore_path}' for to_ignore_path in to_ignore_paths])

    name_to_module = get_name_to_module_path_map(imported_modules, root_path,
                                                 to_ignore_paths)

    code_paths = [
        _get_code_path(module_name, path)
        for module_name, path in name_to_module.items()
    ]
    return list(set(code_paths))
