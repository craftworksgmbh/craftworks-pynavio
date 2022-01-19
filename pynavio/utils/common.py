import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

ExampleRequestType = Optional[Dict[str, List[Dict[str, Any]]]]


def get_module_path(module: ModuleType) -> str:
    """ Use for local (non pip installed) modules only.
    This is useful for trainer models.
    """
    return str(Path(inspect.getfile(module)).parent)


def _get_path_as_str(path_like: Union[str, Path]) -> str:
    path = Path(path_like)
    assert path.exists(), f"{path_like} does not exist"
    return str(path.parent) if path.is_file() else str(path)


def _generate_default_to_ignore_dirs(module_path):
    to_ignore_dirs = [
        path for path in Path(module_path).rglob("*venv*") if path.is_dir()
    ]
    to_ignore_parent_dirs = [
        path for path in Path(module_path).rglob("*site-packages*")
        if path.is_dir()
    ]
    [to_ignore_dirs.append(path.parent) for path in to_ignore_parent_dirs]
    return to_ignore_dirs
