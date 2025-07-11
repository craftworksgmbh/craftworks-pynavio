import jsonschema
import pytest

import pynavio

import platform
import pip


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


PREDICTION_KEY = 'prediction'


@pytest.mark.parametrize("model_output", [({
    PREDICTION_KEY: 5
}), ({
    PREDICTION_KEY: 5.0
}), ({
    PREDICTION_KEY: "test"
}), ({
    PREDICTION_KEY: "bool"
}), ({
    PREDICTION_KEY: []
}), ({
    PREDICTION_KEY: {}
}), ({
    PREDICTION_KEY: [{}]
}), ({
    PREDICTION_KEY: [[1], [2]]
}), ({
    PREDICTION_KEY: [8.5, {}]
}), ({
    PREDICTION_KEY: [8.5, "f"]
}), ({
    PREDICTION_KEY: [8.5, True]
}), ({
    PREDICTION_KEY: ["True", True]
})])
def test_prediction_schema_check_negative(model_output):
    with pytest.raises(jsonschema.exceptions.ValidationError):
        pynavio.mlflow.ModelValidator.verify_model_output(model_output)


@pytest.mark.parametrize("model_output", [
    ({
        PREDICTION_KEY: [5]
    }),
    ({
        PREDICTION_KEY: [5],
        "extra key": 7,
        "other extra": {
            "more info": []
        }
    }),
    ({
        PREDICTION_KEY: [5.1, 5.0]
    }),
    ({
        PREDICTION_KEY: ["a", "b"]
    }),
    ({
        PREDICTION_KEY: [True, False, True]
    }),
])
def test_prediction_schema_check_positive(model_output):
    try:
        pynavio.mlflow.ModelValidator.verify_model_output(model_output)
    except (jsonschema.exceptions.ValidationError, AssertionError):
        pytest.fail("Unexpected Exception")


call_kwargs = {
    'append_to_failed_msg': ' Failed !!!',
    'append_to_succeeded_msg': ' Succeeded !!!'
}


@pytest.mark.parametrize("call_kwargs, msg", [
    ({}, ''),
    (call_kwargs, call_kwargs['append_to_succeeded_msg']),
])
def test_ModelValidator_call_positive(monkeypatch, capfd, call_kwargs, msg):

    def mock_run(self, path, model_zip, model_zip_size_limit, **kwargs):
        pass

    monkeypatch.setattr('pynavio.mlflow.ModelValidator._run', mock_run)
    pynavio.mlflow.ModelValidator()('path/to/model', '', 0, **call_kwargs)
    out, err = capfd.readouterr()
    expected_msg = f"{pynavio.mlflow.pynavio_model_validation}:" \
                   f" Validation succeeded.{msg}\n"
    assert out == expected_msg


@pytest.mark.parametrize("call_kwargs, msg", [
    ({}, ''),
    (call_kwargs, call_kwargs['append_to_failed_msg']),
])
def test_ModelValidator_call_negative(monkeypatch, capfd, call_kwargs, msg):

    def mock_run(self, path, model_zip, model_zip_size_limit, **kwargs):
        assert False

    monkeypatch.setattr('pynavio.mlflow.ModelValidator._run', mock_run)
    with pytest.raises(AssertionError):
        pynavio.mlflow.ModelValidator()('path/to/model', '', 0, **call_kwargs)
    out, err = capfd.readouterr()
    expected_msg = f'{pynavio.mlflow.pynavio_model_validation}: ' \
                   f'Validation failed. Please fix' \
                   f' the identified issues before' \
                   f' uploading the model.{msg}\n'

    assert out == expected_msg


@pytest.mark.parametrize("schema_file_name, is_nested",
                         [('example_request_nested.json', True),
                          ('example_request.json', False)])
def test_is_input_nested(fixtures_path, schema_file_name, is_nested):
    import json
    schema_path = fixtures_path / 'schemas' / schema_file_name

    with open(schema_path, 'r') as schema_file:
        example_request = json.load(schema_file)

    assert pynavio.mlflow.is_input_nested(example_request,
                                          pynavio.mlflow.
                                          not_nested_request_schema())\
           == is_nested


def test__add_sys_dependencies(tmp_path):
    pynavio.mlflow._add_sys_dependencies(tmp_path, ["lib1", "lib2"])
    file_path = tmp_path / 'sys_dependencies.txt'

    with open(file_path, 'r') as result_file:
        file_content = result_file.read()
        assert file_content == "lib1\nlib2"


def test__add_sys_dependencies_fails_on_str(tmp_path):
    with pytest.raises(AssertionError):
        pynavio.mlflow._add_sys_dependencies(tmp_path, "lib1")


def test__add_sys_dependencies_no_resulting_file(tmp_path):
    import os
    pynavio.mlflow._add_sys_dependencies(tmp_path, None)
    file_path = tmp_path / 'sys_dependencies.txt'

    assert not os.path.exists(file_path)


def test__is_wrapped_by_prediction_call():

    def predict():
        pass

    wrapped_predict = pynavio.model_helpers.prediction_call(predict())

    assert pynavio.mlflow._is_wrapped_by_prediction_call(wrapped_predict) \
           is True
    assert pynavio.mlflow._is_wrapped_by_prediction_call(predict) is False


def test_is_model_predict_wrapped_by_prediction_call(tmp_path):

    from pathlib import Path
    from tempfile import TemporaryDirectory
    import mlflow

    import pynavio

    TARGET = 'target'
    _columns = ['x', 'y']
    example_request = pynavio.make_example_request(
        {
            TARGET: float(sum(range(len(_columns)))),
            **{col: float(i) for i, col in enumerate(_columns)}
        },
        target=TARGET)

    class SampleModel(mlflow.pyfunc.PythonModel):

        @pynavio.prediction_call
        def predict(self, context, model_input):
            return {
                'prediction': [1.] * model_input.shape[0]
            }

    def setup(path: Path, *args, **kwargs):
        with TemporaryDirectory():
            pynavio.mlflow.to_navio(SampleModel(),
                                    example_request=example_request,
                                    code_path=kwargs.get('code_path'),
                                    path=path,
                                    pip_packages=['mlflow'])

    model_path = str(tmp_path / 'model')

    setup_arguments = dict(with_data=False,
                           with_oodd=False,
                           explanations=None,
                           path=model_path)

    setup(**setup_arguments)

    model = mlflow.pyfunc.load_model(model_path)
    try:
        pynavio.mlflow._is_model_predict_wrapped_by_prediction_call(model)
    except AttributeError:
        raise pytest.fail(
            "did raise AttributeError, therefore currently prediction call"
            " usage is not being checked")
    except Exception:
        raise pytest.fail(f"did raise {Exception}")


def sample_model(tmp_path, extra_pip_packages, pip_packages, conda_env):
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import mlflow

    import pynavio

    TARGET = 'target'
    _columns = ['x', 'y']
    example_request = pynavio.make_example_request(
        {
            TARGET: float(sum(range(len(_columns)))),
            **{col: float(i) for i, col in enumerate(_columns)}
        },
        target=TARGET)

    class SampleModel(mlflow.pyfunc.PythonModel):

        @pynavio.prediction_call
        def predict(self, context, model_input):
            return {'prediction': [1.] * model_input.shape[0]}

    def setup(path: Path, *args, **kwargs):
        with TemporaryDirectory():
            pynavio.mlflow.to_navio(SampleModel(),
                                    example_request=example_request,
                                    code_path=kwargs.get('code_path'),
                                    conda_env=conda_env,
                                    pip_packages=pip_packages,
                                    extra_pip_packages=extra_pip_packages,
                                    path=path)

    model_path = str(tmp_path / 'model')

    setup_arguments = dict(with_data=False,
                           with_oodd=False,
                           explanations=None,
                           path=model_path)

    return setup(**setup_arguments)


@pytest.mark.parametrize("extra_pip_packages, pip_packages, conda_env",
                         [(['mlflow'], ['numpy'], 'dummy_conda_env'),
                          (['mlflow'], ['numpy'], None),
                          (['mlflow'], None, 'dummy_conda_env'),
                          (None, ['numpy'], 'dummy_conda_env')])
def test_to_navio_extra_dependencies_negative(tmp_path, extra_pip_packages,
                                              pip_packages, conda_env):
    with pytest.raises(AssertionError,
                       match="The arguments 'extra_pip_packages', "
                             "'pip_packages' and 'conda_env' cannot "
                             "be specified at the same time."):
        sample_model(tmp_path, extra_pip_packages, pip_packages, conda_env)


# Create dummy conda env required for the test
dummy_conda_env = conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                f'python={platform.python_version()}',
                f'pip={pip.__version__}', {
                    'pip': ['numpy']
                }
            ],
            'name': 'venv'
        }


@pytest.mark.parametrize("extra_pip_packages, pip_packages, conda_env",
                         [(['mlflow'], None, None),
                          (None, None, dummy_conda_env),
                          (None, ['numpy'], None)
                          ])
def test_to_navio_extra_dependencies(tmp_path, extra_pip_packages,
                                     pip_packages, conda_env):
    try:
        sample_model(tmp_path, extra_pip_packages, pip_packages,
                     conda_env)
    except Exception:
        raise pytest.fail("Unexpected Exception")


@pytest.mark.parametrize("to_navio_kwargs, expected_metadata", [
    (
        {
            "metadata": {"key1": "value1", "key2": "value2"},
            "num_gpus": 2,
        },
        {
            "key1": "value1",
            "key2": "value2",
            "request_schema": {"path": "artifacts/example_request.json"},
            "explanations": "default",
            "oodDetection": "default",
            "gpus": 2,
        }
    ),
    (
        {
            "metadata": None,
            "num_gpus": 1,
        },
        {
            "request_schema": {"path": "artifacts/example_request.json"},
            "explanations": "default",
            "oodDetection": "default",
            "gpus": 1,
        }
    )
])
def test_to_navio_metadata(tmp_path, to_navio_kwargs, expected_metadata):
    import yaml
    import mlflow
    import pynavio

    from tempfile import TemporaryDirectory
    from pathlib import Path

    # Specify example request
    TARGET = 'target'
    _columns = ['x', 'y']
    example_request = pynavio.make_example_request(
        data={
            TARGET: float(sum(range(len(_columns)))),
            **{col: float(i) for i, col in enumerate(_columns)}
        },
        target=TARGET
    )

    # Create dummy model
    class SampleModel(mlflow.pyfunc.PythonModel):

        @pynavio.prediction_call
        def predict(self, context, model_input):
            return {
                'prediction': [1.] * model_input.shape[0]
            }

    # Set model path
    model_path = str(tmp_path / 'model')

    # Setup model
    with TemporaryDirectory():
        pynavio.mlflow.to_navio(
            model=SampleModel(),
            path=model_path,
            example_request=example_request,
            pip_packages=['mlflow'],
            **to_navio_kwargs,

        )

        path = Path(model_path) / 'MLmodel'
        with path.open('r') as file:
            cfg = yaml.safe_load(file)

    # Assert correct metadata
    assert cfg["metadata"] == expected_metadata
