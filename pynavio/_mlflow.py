import json
import shutil
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union
import subprocess
import time
import requests
from collections.abc import Mapping

import mlflow
import copy
import pandas as pd
import yaml
import jsonschema

from pynavio.utils import ExampleRequestType, make_env
from pynavio.utils.json_encoder import JSONEncoder
from pynavio.utils.schemas import PREDICTION_SCHEMA, \
    METADATA_SCHEMA, REQUEST_SCHEMA_SCHEMA

EXAMPLE_REQUEST = 'example_request'
ARTIFACTS = 'artifacts'
ArtifactsType = Optional[Dict[str, str]]
ERROR_KEYS = {'error_code', 'message', 'stack_trace'}
PREDICTION_KEY = 'prediction'


def _get_field(yml: dict, path: str) -> Optional[Any]:
    keys = path.split('.')
    assert keys, 'Path must not be empty'

    value = yml.get(keys[0])
    for key in keys[1:]:
        if not isinstance(value, dict):
            return None
        value = value.get(key)

    return value


def register_example_request(
        tmp_dir,
        example_request: ExampleRequestType = None,
        artifacts: ArtifactsType = None) -> Dict[str, str]:
    """
    @param tmp_dir: temporary directory
    @param example_request: example_request for the given model.
     If not set, needs to be present in artifacts.
    @param artifacts:If not set, need to set example_request
    @return: artifacts containing example request
    """
    assert any(item is not None for item in [example_request, artifacts]),\
        f"either {EXAMPLE_REQUEST} or {ARTIFACTS} need to be set"

    if example_request:
        # add example_request to artifacts
        artifacts = {
            EXAMPLE_REQUEST: f'{tmp_dir}/{EXAMPLE_REQUEST}.json',
            **(artifacts or {})
        }
        with open(artifacts[EXAMPLE_REQUEST], 'w') as file:
            json.dump(example_request, file, indent=4, cls=JSONEncoder)
    else:
        # make sure example_request already exists in the artifacts
        assert EXAMPLE_REQUEST in artifacts, f'if {EXAMPLE_REQUEST} ' \
                                             f'argument is not set,' \
                                             f' it needs to be present' \
                                             f' in {ARTIFACTS}'
        assert Path(artifacts[EXAMPLE_REQUEST]).exists()

    return artifacts


def _safe_code_path(code_path: Union[List[Union[str, PosixPath]], None]):
    if code_path is not None:
        assert all(Path(p).is_dir() for p in code_path), \
            'All code dependencies must be directories'
        assert not any(Path(p).resolve().absolute() == Path.cwd().absolute()
                       for p in code_path), \
            'Code paths must not contain the current directory'
        # deleting __pycache__, otherwise MLFlow adds it to the code directory
        for path in code_path:
            for cache_dir in Path(path).glob('**/__pycache__'):
                shutil.rmtree(cache_dir, ignore_errors=True)
    else:
        code_path = None
    return code_path


def _check_data_spec(spec: dict) -> None:
    for field in ['name', 'path']:
        assert field in spec, f'Data spec is missing field {field}'
        assert isinstance(spec[field], str), \
            f'Expected field {field} in data spec to be of type str, got' \
            f'{type(spec[field])}'


def _add_metadata(model_path: str,
                  dataset: Optional[dict] = None,
                  explanations: Optional[str] = None,
                  oodd: Optional[str] = None,
                  num_gpus: Optional[int] = 0) -> None:
    path = Path(model_path) / 'MLmodel'
    with path.open('r') as file:
        cfg = yaml.safe_load(file)
    example_request_path_yml = 'flavors.python_function.artifacts.' \
                               'example_request.path'
    cfg.update(metadata=dict(request_schema=dict(
        path=_get_field(cfg, example_request_path_yml))))

    if dataset is not None:
        dataset_path_yml = 'flavors.python_function.artifacts.dataset.path'
        cfg['metadata'].update(dataset=dataset)
        cfg['metadata']['dataset']['path'] = _get_field(cfg, dataset_path_yml)

    explanations = explanations or 'default'
    accepted_values = ['disabled', 'default', 'plotly']
    assert explanations in accepted_values, \
        f'explanations config must be one of {accepted_values}'
    cfg['metadata'].update(explanations=explanations)

    oodd = oodd or 'default'
    accepted_values = ['disabled', 'default']
    assert oodd in accepted_values, \
        f'oodd config must be one of {accepted_values}'
    cfg['metadata'].update(oodDetection=oodd)

    assert num_gpus >= 0, 'num_gpus cannot be negative'
    if num_gpus > 0:
        cfg['metadata'].update(gpus=num_gpus)

    with path.open('w') as file:
        yaml.dump(cfg, file)


ExampleRequest = Dict[str, List[Dict[str, Any]]]


def process_path(path):
    str_path = str(path)
    str_path = str_path[7:] if str_path[0:7] == 'file://' else str_path
    return str_path


def _read_mlmodel_yaml(model_path):
    with (Path(model_path) / 'MLmodel').open('r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def _read_example_request(model_path, config):
    schema_path = Path(model_path) / _get_field(
        config, 'metadata.request_schema.path')

    with open(schema_path, 'r') as schema_file:
        schema = json.load(schema_file)
    return schema


def _read_metadata(model_path: str) -> dict:
    yml = _read_mlmodel_yaml(model_path)

    data_path = _get_field(yml, 'metadata.dataset.path')
    data_path = Path(model_path) / data_path if data_path is not None else None

    example_request = _read_example_request(model_path, yml)

    return {
        'dataset':
            pd.read_csv(data_path) if data_path is not None else None,
        'explanation_format':
            _get_field(yml, 'metadata.explanations.format'),
        'example_request':
            example_request
    }


def _fetch_data(model_path: str) -> dict:
    meta = _read_metadata(model_path)
    data = meta['example_request']

    _input = {
        'columns': [x['name'] for x in data['featureColumns']],
        'data': [[x['sampleData'] for x in data['featureColumns']]]
    }

    if 'dateTimeColumn' in data:
        _input['columns'].append(data['dateTimeColumn']['name'])
        _input['data'][0].append(data['dateTimeColumn']['sampleData'])

    if meta.get('explanation_format') in ['default', None]:
        return [_input]

    dataset = meta.get('dataset')
    if dataset is None:
        return [_input]

    _explain_input = copy.deepcopy(_input)
    _explain_input['columns'].append('is_background')
    _explain_input['data'] = [[
        *_explain_input['data'][0], False
    ], *dataset[_input['columns']].assign(is_background=True).sample(
        20, random_state=42, replace=True).values.tolist()]

    return [_input, _explain_input]


def _get_example_request_df(model_path):
    data = _fetch_data(model_path)[0]
    return pd.DataFrame(data['data'], columns=data['columns'])


class _ModelValidator:
    @staticmethod
    def validate_metadata(model_path):
        config = _read_mlmodel_yaml(model_path)
        try:
            jsonschema.validate(config.get('metadata'), METADATA_SCHEMA)
        except jsonschema.exceptions.ValidationError:
            print("Error during MLmodel validation")
            raise

        example_request = _read_example_request(model_path, config)
        try:
            jsonschema.validate(example_request, REQUEST_SCHEMA_SCHEMA)
        except jsonschema.exceptions.ValidationError:
            print("Error during example request validation")
            raise

    @staticmethod
    def run_model_io(model_path, model_input=None, **kwargs):
        model = mlflow.pyfunc.load_model(model_path)
        if model_input is None:
            model_input = _get_example_request_df(model_path)
        return model_input, model.predict(model_input)

    @staticmethod
    def _check_if_prediction_call_is_used(model_path):
        # In order to check if the pynavio.prediction_call
        # is used to decorate the predict method of the model,
        # a corrupt input is provided. If there is an exception
        # or the expected keys are not present in the model output,
        # then that means that the decorator is not applied

        model = mlflow.pyfunc.load_model(model_path)
        corrupt_model_input = pd.DataFrame(
            {'corrupt_model_input_123': [1, 2, 3]})

        import logging
        logger = logging.getLogger('gunicorn.error')
        current_loglevel = logger.level
        logger.setLevel(logging.CRITICAL)  # this is done to prevent
        # logging the exception traceback from prediction call
        # as it will be confusing for the user

        error_msg = \
            "Please use pynavio.prediction_call to decorate " \
            "the predict method of the model, which will add the " \
            "needed error keys for error case"
        try:
            model_output = model.predict(corrupt_model_input)
        except Exception:
            raise KeyError(error_msg)
        finally:
            logger.setLevel(current_loglevel)

        assert isinstance(model_output, Mapping), "Model " \
            "output has to be a dictionary"

        if PREDICTION_KEY not in model_output:  # precaution
            # to allow for models that always return prediction

            assert set(model_output.keys()) == ERROR_KEYS, error_msg

    @staticmethod
    def verify_model_output(model_output, **kwargs):
        def _validate_prediction_schema(model_prediction):
            try:
                jsonschema.validate(model_prediction, PREDICTION_SCHEMA)
            except jsonschema.exceptions.ValidationError:
                print(f"Error: The value of model_output['{PREDICTION_KEY}']"
                      " must be one of the following types "
                      "(cannot be nested or mixed type): "
                      "'array','boolean', 'integer', 'number', 'string'")
                raise

        assert isinstance(model_output, Mapping), "Model " \
            "output has to be a dictionary"

        if PREDICTION_KEY in model_output:
            _validate_prediction_schema(model_output)
        else:
            assert set(model_output.keys()) == ERROR_KEYS, \
                f"The model output has to contain '{PREDICTION_KEY}'" \
                f" for prediction" \
                f" as key for the target, independent of" \
                f" the target name in the example request" \
                f". There can be other keys, " \
                f" that will be listed under " \
                f" 'additionalFields' in the response of the model " \
                f"deployed to navio" \
                " example model output: {'prediction': " \
                "[1.] * model_input.shape[0],"\
                " 'extra': { 'this': 'can be any JSON serializable" \
                " structure',} }"\
                f" in the response of the model deployed" \
                f" to navio."\
                f" Or The model output has to contain the following" \
                f" keys [{ERROR_KEYS}] if error occurs."\
                f"Please use pynavio.prediction_call to decorate " \
                f"the predict method of the model, which will add the " \
                f"needed error keys for error case"

    def __call__(self, model_path, **kwargs):
        self.validate_metadata(model_path)
        model_input, model_output = self.run_model_io(model_path)
        self._check_if_prediction_call_is_used(model_path)
        self.verify_model_output(model_output)


def check_model_serving(model_path: Union[str, Path],
                        port=5001, request_bodies=None):
    """
    checks model serving with mlflow. This has limitations, e.g.
    the 'conda.env' setup will not be checked.
    Note: Please refer to
    https://navio.craftworks.io/docs/guides/navio-models/model_creation/#3-test-model-serving
    for testing the model serving.

    @param model_path: model path
    @param port: port to use for model serving, defaults to 5001
    @param request_bodies: request bodies to use
     for checking the model serving, defaults to
     using the example request from the model

    Will throw an exception if check does not pass.
    """
    URL = f'http://127.0.0.1:{port}/invocations'
    process = subprocess.Popen(
        f'mlflow models serve -m {model_path} -p {port} --no-conda'.split())
    time.sleep(5)
    response = None

    try:
        for data in (request_bodies or _fetch_data(model_path)):
            response = requests.post(
                URL,
                data=json.dumps(data, allow_nan=True),
                headers={'Content-type': 'application/json'})
            response.raise_for_status()
    finally:
        process.terminate()
        if response is not None:
            print(response.json())
        subprocess.run('pkill -f gunicorn'.split())
        time.sleep(2)


def to_navio(model: mlflow.pyfunc.PythonModel,
             path,
             example_request: ExampleRequestType = None,
             pip_packages: List[str] = None,
             code_path: Optional[List[Union[str, Path]]] = None,
             conda_packages: List[str] = None,
             conda_channels: List[str] = None,
             conda_env: str = None,
             artifacts: ArtifactsType = None,
             dataset: Optional[dict] = None,
             explanations: Optional[str] = None,
             oodd: Optional[str] = None,
             num_gpus: Optional[int] = 0
             ) -> Path:
    """
    create a .zip mlflow model file for navio
    Usage: either pip_packages or conda_env need to be set.

    @param model: model to save
    @param path: path of where model .zip file needs to be saved
    @param example_request: example_request for the given model.
    If not set, needs to be present in artifacts.
    @param pip_packages: list of pip packages(optionally with versions)
    with the syntax of a requirements.txt file, e.g.
    ['mlflow==1.15.0', 'scikit_learn == 0.24.1'].
    Tip: For most cases it should be enough to use
    pynavio.utils.infer_dependencies.infer_external_dependencies().
    @param code_path: A list of local filesystem paths to Python file
    dependencies (or directories containing file dependencies)
    @param conda_packages: list of conda packages
    @param conda_channels: list of conda channels
    @param conda_env: the path of a conda.yaml file to use. If specified,
    the values of conda_channels, conda_packages and pip_packages would be
    ignored.
    @param artifacts: If not set, need to set example_request
    @param dataset:
    @param explanations:
    @param oodd:
    @param num_gpus:

    Note: Please refer to check_model_serving() method and
    https://navio.craftworks.io/docs/guides/navio-models/model_creation/#3-test-model-serving
    for testing the model serving.

    @return: path to the .zip model file
    """
    path = process_path(path)

    if code_path:
        if not isinstance(code_path, list):
            raise TypeError("'code_path' argument must be a"
                            " list (of local filesystem paths to Python file"
                            "dependencies (or directories containing file "
                            "dependencies)), but is not a list")

        if any(Path(code_p).resolve() in Path(path).resolve().parents
               for code_p in code_path):
            raise ValueError("any of 'code_path' argument paths cannot"
                             " be a parent of 'path' argument,"
                             f" please change the 'path': {path} to be"
                             f" outside of the 'code_path' paths:{code_path}")

    artifacts = artifacts or dict()
    artifacts = {key: process_path(value) for key, value in artifacts.items()}

    with TemporaryDirectory() as tmp_dir:
        if dataset is not None:
            _check_data_spec(dataset)
            artifacts.update(dataset=dataset['path'])

        conda_env = make_env(pip_packages, conda_packages, conda_channels,
                             conda_env)

        code_path = _safe_code_path(code_path)

        artifacts = register_example_request(tmp_dir, example_request,
                                             artifacts)

        shutil.rmtree(path, ignore_errors=True)
        mlflow.pyfunc.save_model(path=path,
                                 python_model=model,
                                 conda_env=conda_env,
                                 artifacts=artifacts,
                                 code_path=code_path)

        _add_metadata(path,
                      dataset=dataset,
                      explanations=explanations,
                      oodd=oodd,
                      num_gpus=num_gpus)

    _ModelValidator()(path)

    shutil.make_archive(path, 'zip', path)
    return Path(path + '.zip')
