import platform

import pip
import pytest
from pynavio.mlflow import ARTIFACTS, register_example_request
from pynavio.utils import make_env


def test_make_conda_env_negative_wrong_arguments():
    with pytest.raises(Exception):
        make_env()


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
])
def test_make_conda_env_positive(args, expected):
    conda_env = make_env(**args)
    assert conda_env == expected


@pytest.mark.parametrize(
    "args",
    [
        # both arguments not set
        (dict()),
        # artifacts is set, but does not contain example_request
        ({
            ARTIFACTS: {
                'model_path': "a_path"
            }
        }),
        # artifacts is set, example_request file path does not exist
        ({
            ARTIFACTS: {
                'example_request': "non_existent_path"
            }
        })
    ])
def testregister_example_request_negative(args, tmpdir):
    with pytest.raises(AssertionError):
        register_example_request(tmpdir, **args)
