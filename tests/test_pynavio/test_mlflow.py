import pytest

from pynavio.mlflow import ARTIFACTS, register_example_request


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
