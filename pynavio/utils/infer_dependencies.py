import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pkg_resources


def _generate_requirements_txt_file(requirements_txt_file, module_path):
    yes = subprocess.Popen(('yes', 'N'), stdout=subprocess.PIPE)
    assert Path(module_path).exists(), f"{module_path} does not exist"
    result = subprocess.call(
        ('pigar', '-P', f'{module_path}', '-p', f'{requirements_txt_file}',
         '--without-referenced-comments'),
        stdin=yes.stdout)
    if result != 0:
        logging.error(f"please create and provide requirements.txt, as there was an error using pigar" \
                      f" to auto-generate requirements.txt")
        raise AssertionError


def infer_external_dependencies(module_path: str) -> List[str]:
    """
    infers pip requirement strings.
    known edge cases and limitations:
     - in case of some libs, e.g. for pytorch, installing via pip is not recommended when using conda
    and would result in a broken conda env
     -it might not be able to detect all the required dependencies,
     in which case the user could append/extend the list manually
    @param module_path:
    @return: list of inferred pip requirements, e.g. ['mlflow==1.15.0', 'scikit_learn == 0.24.1']
    """
    with TemporaryDirectory() as tmp_dir:
        requirements_txt_file = Path(tmp_dir) / 'requirements.txt'
        _generate_requirements_txt_file(requirements_txt_file, module_path)
        requirements = read_requirements_txt(requirements_txt_file)
    return requirements


def read_requirements_txt(requirements_txt_path) -> list:

    with Path(requirements_txt_path).open() as requirements_txt:
        requirements = [
            str(requirement) for requirement in
            pkg_resources.parse_requirements(requirements_txt)
        ]
    return requirements
