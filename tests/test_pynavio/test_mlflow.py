import jsonschema
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
def test_is_input_nested(rootpath, schema_file_name, is_nested):
    import json
    schema_path = rootpath / \
        'tests'/'test_pynavio'/'fixtures'/'schemas'/schema_file_name

    with open(schema_path, 'r') as schema_file:
        example_request = json.load(schema_file)

    assert pynavio.mlflow.is_input_nested(example_request,
                                          pynavio.mlflow.
                                          not_nested_request_schema())\
           == is_nested


def test__add_sys_dependencies():
    import os
    dep_path = "."
    pynavio.mlflow._add_sys_dependencies(dep_path, ["lib1", "lib2"])

    with open("sys_dependencies.txt", 'r') as result_file:
        file_content = result_file.read()
        # cleanup
        os.remove(f"{dep_path}/sys_dependencies.txt")
        assert file_content == "lib1\nlib2"


def test__add_sys_dependencies_fails_on_str():
    with pytest.raises(AssertionError):
        pynavio.mlflow._add_sys_dependencies("", "lib1")


def test__add_sys_dependencies_no_resulting_file():
    import os
    dep_path = ""
    pynavio.mlflow._add_sys_dependencies(dep_path, None)

    assert not os.path.exists("sys_dependencies.txt")


def test__is_wrapped_by_prediction_call():

    def predict():
        pass

    wrapped_predict = pynavio.model_helpers.prediction_call(predict())

    assert pynavio.mlflow._is_wrapped_by_prediction_call(wrapped_predict) \
           is True
    assert pynavio.mlflow._is_wrapped_by_prediction_call(predict) is False


def test_is_model_predict_wrapped_by_prediction_call(tmp_path):
    import mlflow

    from examples import mlflow_models
    from examples.mlflow_models import tabular
    model_path = str(tmp_path / 'model')

    setup_arguments = dict(with_data=False,
                           with_oodd=False,
                           explanations=None,
                           path=model_path,
                           code_path=[mlflow_models.__path__[0]])

    tabular.setup(**setup_arguments)

    model = mlflow.pyfunc.load_model(model_path)
    try:
        pynavio.mlflow._is_model_predict_wrapped_by_prediction_call(model)
    except AttributeError:
        raise pytest.fail(
            "did raise AttributeError, therefore currently prediction call"
            " usage is not being checked"
        )
    except Exception:
        raise pytest.fail(f"did raise {Exception}")
