import logging
import subprocess
import pkg_resources
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Union

from .utils.common import _generate_default_to_ignore_dirs, _get_path_as_str


def _generate_ignore_dirs_args(module_path, to_ignore_dirs):
    ignore_dirs_args = []
    if to_ignore_dirs is None:
        to_ignore_dirs = _generate_default_to_ignore_dirs(module_path)
    else:
        for ignore_dir in to_ignore_dirs:
            assert Path(ignore_dir).exists(), f"{module_path} does not exist"
    if to_ignore_dirs:
        ignore_dirs_args = ['-i', *to_ignore_dirs]
    return ignore_dirs_args


def _generate_requirements_txt_file(requirements_txt_file,
                                    module_path: Union[str, Path],
                                    to_ignore_dirs=None):
    yes = subprocess.Popen(('yes', 'N'), stdout=subprocess.PIPE)

    module_path = _get_path_as_str(module_path)
    ignore_dirs_args = _generate_ignore_dirs_args(module_path, to_ignore_dirs)

    result = subprocess.call(
        ('pigar', '-P', f'{module_path}', '-p', f'{requirements_txt_file}',
         *ignore_dirs_args, '--without-referenced-comments'),
        stdin=yes.stdout)
    if result != 0:
        logging.error("please create and provide requirements.txt, as "
                      "there was an error using pigar to auto-generate "
                      "requirements.txt")
        raise AssertionError


def read_requirements_txt(requirements_txt_path) -> list:

    with Path(requirements_txt_path).open() as requirements_txt:
        requirements = [
            str(requirement) for requirement in
            pkg_resources.parse_requirements(requirements_txt)
        ]
    return requirements


def infer_external_dependencies(
        module_path: Union[str, Path],
        to_ignore_paths: List[str] = None) -> List[str]:
    """
    infers pip requirement strings.
    known edge cases and limitations:
     - in case of some libs, e.g. for pytorch, installing via pip is not
     recommended when using conda
    and would result in a broken conda env
     - it might add packages, that are not being used ( e.g. import
     statements under conditional operators, with false condition)
     - it might not be able to detect all the required dependencies,
     in which case the user could append/extend the list manually
    @param module_path:
    @param to_ignore_paths: list of paths to ignore.
     -Ignores a directory named *venv* or containing *site-packages* by
     default
    @return: list of inferred pip requirements, e.g.
    ['mlflow==1.15.0', 'scikit_learn == 0.24.1']
    """
    with TemporaryDirectory() as tmp_dir:
        requirements_txt_file = Path(tmp_dir) / 'requirements.txt'
        _generate_requirements_txt_file(requirements_txt_file, module_path,
                                        to_ignore_paths)
        requirements = read_requirements_txt(requirements_txt_file)
    return requirements
