import platform
from typing import Any, Dict, List

import pip


def make_env(
    pip_packages: List[str] = None,
    conda_packages: List[str] = None,
    conda_channels: List[str] = None,
    conda_env: str = None,
) -> Dict[str, Any]:
    """
    makes the value for the mlflow.pyfunc.save_model()'s conda_env argument
    Usage: either pip_packages or conda_env need to be set.
    @param pip_packages:
    @param conda_packages:
    @param conda_channels:
    @param conda_env: the path of a conda.yaml file to use. If specified,
    the values of conda_channels, conda_packages and pip_packages would be
    ignored.
    @return:
    """
    assert any(item is not None for item in [pip_packages, conda_env]),\
        "either 'pip_packages' or 'conda_env' need to be set"

    if conda_env is None:
        conda_env = {
            'channels': ['defaults', 'conda-forge', *(conda_channels or [])],
            'dependencies': [
                f'python={platform.python_version()}',
                f'pip={pip.__version__}', *(conda_packages or []), {
                    'pip': pip_packages
                }
            ],
            'name': 'venv'
        }

    return conda_env
