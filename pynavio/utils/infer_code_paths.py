import inspect
import sys
from pathlib import Path
from typing import Dict, List
from pigar.parser import Module

from .common import get_module_path


def _get_code_path(module_name: str, path: str) -> List[str]:
    """
    get code path of the module name that is highest in the import hierarchy
    (e.g. for pynavio.utils it will be the path of pynavio)
    @param module_name: import string of the module
    @param path: module path
    @return: code path of the module name (highest in the import hierarchy)
     - Note: returns an empty string if path is not found
    """

    module_base = next(part for part in module_name.split(".") if part)
    if sys.modules.get(module_base):
        if getattr(sys.modules.get(module_base), '__path__', False):
            path = sys.modules.get(module_base).__path__[0]
        else:
            path = ''
    else:
        # fallback solution:
        # in case the there are more than one occurrences
        # of a directory with the module name,
        # the returned path will be the last occurrence
        path_parts = Path(path).parts
        index_of_folder_name_in_the_path = next(
            i for i in reversed(range(len(path_parts)))
            if path_parts[i] == module_base)
        path = Path(*path_parts[:index_of_folder_name_in_the_path + 1])
    return f'{path}'


def get_name_to_module_path_map(imported_modules: List[Module], root_path: str,
                                to_ignore_paths: List[str]) -> Dict[str, str]:

    name_to_module_path = dict()
    for module in imported_modules:
        name = module.name
        sys_module_obj = sys.modules.get(name, None)

        if inspect.ismodule(sys_module_obj) and getattr(
                sys_module_obj, '__file__', None):

            if Path(root_path) in Path(get_module_path(
                    sys_module_obj)).parents and _is_not_in_ignore_paths(
                        sys_module_obj, to_ignore_paths):
                name_to_module_path[module.name] = get_module_path(
                    sys_module_obj)
    return name_to_module_path


def _is_not_in_ignore_paths(module, to_ignore_paths):
    return not any([
        Path(to_ignore_path) in Path(get_module_path(module)).parents
        for to_ignore_path in to_ignore_paths
    ])


