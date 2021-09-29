import platform

import pip
import pytest

from pynavio.utils.common import (ARTIFACTS, EXAMPLE_REQUEST,
                                  _add_example_request_artifact,
                                  _make_conda_env)


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
    conda_env = _make_conda_env(**args)
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
def test_add_example_request_artifact_negative(args, tmpdir):
    with pytest.raises(AssertionError):
        _add_example_request_artifact(tmpdir, **args)
