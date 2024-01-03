import platform

import pip
import pytest

from pynavio.utils import make_env

yaml_path = 'conda.yaml'


@pytest.mark.parametrize("args", [
    ({
        'conda_env': yaml_path
    })
])
def test_make_conda_env_positive_yaml(args, expected=yaml_path):
    conda_env = make_env(**args)
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
        'pip_packages': ['numpy==1.20.2', 'mlflow']
    }, {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            f'python={platform.python_version()}', f'pip={pip.__version__}', {
                'pip': ['numpy==1.20.2', 'mlflow']
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
    ({
        'pip_packages': None,
        'conda_env': None
    }, None)
])
def test_make_conda_env_positive(args, expected):
    conda_env = make_env(**args)
    assert conda_env == expected


@pytest.mark.parametrize("args", [
    ({
         'pip_packages': ['numpy'],
         'conda_env': 'dummy_conda_env'
     })
])
def test_make_conda_env_negative(args):
    with pytest.raises(AssertionError,
                       match="The arguments 'conda_env' and 'pip_packages' "
                             "cannot be specified at the same time"):
        make_env(**args)
