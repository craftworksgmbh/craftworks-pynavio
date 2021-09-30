import inspect
import re
import sys
import types
from pathlib import Path
from typing import List

from pynavio.utils.common import get_module_path
from pynavio.utils.directory_utils import _generate_default_to_ignore_dirs


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

    from pigar.parser import parse_imports
    assert Path(path).exists(), f"{path} does not exist"
    if to_ignore_paths is None:
        to_ignore_paths = _generate_default_to_ignore_dirs(root_path)
    imported_modules, _ = parse_imports(
        path,
        ignores=[f'{to_ignore_path}' for to_ignore_path in to_ignore_paths])
    name_to_module = {}

    for module in imported_modules:
        name = module.name
        moduleobj = sys.modules.get(name, None)

        if inspect.ismodule(moduleobj) and getattr(moduleobj, '__file__',
                                                   None):
            if Path(root_path) in Path(get_module_path(
                    moduleobj)).parents and _is_not_in_ignore_paths(
                        moduleobj, to_ignore_paths):
                name_to_module[module.name] = get_module_path(moduleobj)

    code_paths = []
    #TODO: move to another function & refactor naming
    for module_name, m in name_to_module.items():
        match = re.match('(\.*[^.]+).*', module_name)
        if match:
            n = match.groups()[0]
        else:
            n = module_name
        path_parts = Path(m).parts
        # TODO: improve getting the path or document that will return the last occurrence of module name in the path
        index_of_folder_name_in_the_path = next(
            i for i in reversed(range(len(path_parts))) if path_parts[i] == n)
        code_paths.append(
            f'{Path( *path_parts[:index_of_folder_name_in_the_path+1])}')
    return list(set(code_paths))
