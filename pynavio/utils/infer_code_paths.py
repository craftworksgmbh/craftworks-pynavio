import inspect
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from pigar.parser import Module, parse_imports

from pynavio.utils.common import get_module_path
from pynavio.utils.directory_utils import _generate_default_to_ignore_dirs

RE_FIRST_MODULE_NAME = re.compile('(\.*[^.]+).*')


def _get_module_base(module_name: str) -> str:
    match = re.match(RE_FIRST_MODULE_NAME, module_name)
    if match:
        module_base = match.groups()[0]
    else:
        module_base = module_name
    return module_base


def _get_code_path(module_name: str, path: str) -> List[str]:
    """
    get code path of the module name that is highest in the import hierarchy (e.g. for pynavio.utils it will be the path of pynavio)
    @param module_name: import string of the module
    @param path: module path
    @return: code path of the module name (highest in the import hierarchy)
    """

    module_base = _get_module_base(module_name)
    if sys.modules.get(module_base):
        path = sys.modules.get(module_base).__path__[0]
    else:
        # fallback solution:
        # in case the there are more than one occurrences of a directory with the module name,
        # the returned path will be the last occurrence
        path_parts = Path(path).parts
        index_of_folder_name_in_the_path = next(
            i for i in reversed(range(len(path_parts)))
            if path_parts[i] == module_base)
        path = Path(*path_parts[:index_of_folder_name_in_the_path + 1])
    return f'{path}'


def get_name_to_module_path_map(imported_modules: List[Module], root_path: str, to_ignore_paths: List[str]) ->\
    Dict[str, str]:
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


def infer_imported_code_path(
        path: str,
        root_path: str,
        to_ignore_paths: Optional[List[str]] = None) -> List[str]:
    """
    known edge cases and limitations:
     - Can result in duplicated copies in code_paths if the the imports are inconsistent,
    # e.g. in one place from pynavio.utils.common import get_module_path and in other place
    # from utils.common import get_module_path (with adding more paths to PYTHONPATH)
    @param path: path of the module/file from which to infer the imported code paths
    @param root_path: root path
    @param to_ignore_paths:  list of paths to ignore.
     - Ignores a directory named *venv* or containing *site-packages* by default
    @return: list of imported code paths
    """
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
