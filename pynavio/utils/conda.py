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
    Usage: pip_packages and conda_env are mutually exclusive and cannot be
    set simultaneously. If neither is set, the environment will be inferred
    by mlflow.

    @param pip_packages:
    @param conda_packages:
    @param conda_channels:
    @param conda_env: the path of a conda.yaml file to use. If specified,
    the values of conda_channels, conda_packages and pip_packages would be
    ignored.
    @return:
    """
    assert sum(x is not None for x in [pip_packages,
                                       conda_env]) <= 1, \
        "The arguments 'conda_env' and 'pip_packages' cannot " \
        "be specified at the same time"

    if conda_env is None and pip_packages is None:
        return None
    elif conda_env is None:
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
