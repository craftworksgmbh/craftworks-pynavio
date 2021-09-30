from pathlib import Path


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
