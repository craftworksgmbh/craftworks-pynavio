import pytest

import pynavio


@pytest.mark.parametrize(
    "args",
    [
        # both arguments not set
        (dict()),
        # artifacts is set, but does not contain example_request
        ({
            pynavio.mlflow.ARTIFACTS: {
                'model_path': "a_path"
            }
        }),
        # artifacts is set, example_request file path does not exist
        ({
            pynavio.mlflow.ARTIFACTS: {
                'example_request': "non_existent_path"
            }
        })
    ])
def test_register_example_request_negative(args, tmpdir):
    with pytest.raises(AssertionError):
        pynavio.mlflow.register_example_request(tmpdir, **args)
