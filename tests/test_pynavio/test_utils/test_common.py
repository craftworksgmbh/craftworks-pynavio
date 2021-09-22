import platform

import pip
import pytest

from pynavio.utils.common import _make_conda_env


def test_make_conda_env_negative_wrong_arguments():
    with pytest.raises(Exception):
        _make_conda_env()


yaml_path = 'conda.yaml'


@pytest.mark.parametrize("args", [
    ({
        'conda_env': yaml_path
    }),
    ({
        'conda_env': yaml_path,
        'pip_packages': ['pandas']
    }),
])
def test_make_conda_env_positive_yaml(args, expected=yaml_path):
    conda_env = _make_conda_env(**args)
    assert conda_env == expected


@pytest.mark.parametrize("args, expected", [
    ({
        'pip_packages': ['numpy', 'mlflow']
    }, {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            f'python={platform.python_version()}', f'pip={pip.__version__}', {
                'pip': ['numpy', 'mlflow']
            }
        ],
        'name': 'venv'
    }),
    ({
        'pip_packages': ['numpy', 'mlflow'],
        'conda_channels': ['pytorch']
    }, {
        'channels': ['defaults', 'conda-forge', 'pytorch'],
        'dependencies': [
            f'python={platform.python_version()}', f'pip={pip.__version__}', {
                'pip': ['numpy', 'mlflow']
            }
        ],
        'name': 'venv'
    }),
])
def test_make_conda_env_positive(args, expected):
    conda_env = _make_conda_env(**args)
    assert conda_env == expected
