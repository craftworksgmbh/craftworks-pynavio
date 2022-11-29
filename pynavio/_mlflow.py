import json
import shutil
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union
from collections.abc import Mapping

import mlflow
import copy
import pandas as pd
import yaml
import jsonschema

from pynavio.utils import ExampleRequestType, make_env
from pynavio.utils.json_encoder import JSONEncoder

EXAMPLE_REQUEST = 'example_request'
ARTIFACTS = 'artifacts'
ArtifactsType = Optional[Dict[str, str]]


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


def _read_metadata(model_path: str) -> dict:
    with (Path(model_path) / 'MLmodel').open('r') as file:
        yml = yaml.safe_load(file)

    data_path = _get_field(yml, 'metadata.dataset.path')
    data_path = Path(model_path) / data_path if data_path is not None else None
    example_request_path = Path(model_path) / _get_field(
        yml, 'metadata.request_schema.path')
    with open(example_request_path, 'r') as file:
        example_request = json.load(file)

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
    def run_model_io(model_path, model_input=None, **kwargs):
        model = mlflow.pyfunc.load_model(model_path)
        if model_input is None:
            model_input = _get_example_request_df(model_path)
        return model_input, model.predict(model_input)

    @staticmethod
    def verify_model_output(model_output,
                            expect_error=False, **kwargs):
        def _validate_prediction_schema(model_prediction):
            prediction_types = [
                    'boolean', 'integer', 'number', 'array', 'string'
                ]
            prediction_schema = {
                "type": "object",
                "properties": {
                    prediction_key: {"type": prediction_types},
                },
                "required": [prediction_key],
            }

            try:
                jsonschema.validate(model_prediction, prediction_schema)
            except jsonschema.exceptions.ValidationError:
                print(f"Error: The value of model_output['{prediction_key}']"
                      f" the must be one of the following types: "
                      f"{prediction_types}")
                raise

        assert isinstance(model_output, Mapping), "Model " \
            "output has to be a dictionary"

        if expect_error:
            expected_keys = {'error_code', 'message', 'stack_trace'}
            assert set(model_output.keys()) == expected_keys, \
                "please use pynavio.prediction_call to decorate " \
                "the predict method of the model"

        prediction_key = 'prediction'
        assert prediction_key in model_output, "model output must have" \
               " 'prediction' as key for the target, independent of" \
               " tha target name in the example request. There can be " \
               "other keys, that will be listed under 'additionalFields'" \
               "in the response of the model deployed to navio"
        _validate_prediction_schema(model_output)

    def __call__(self, model_path, expect_error: bool = False, **kwargs):
        model_input, model_output = self.run_model_io(model_path)
        self.verify_model_output(model_output, expect_error=expect_error)


def to_navio(model: mlflow.pyfunc.PythonModel,
             path,
             example_request: ExampleRequestType = None,
             pip_packages: List[str] = None,
             code_path: Optional[List[Union[str, PosixPath]]] = None,
             conda_packages: List[str] = None,
             conda_channels: List[str] = None,
             conda_env: str = None,
             artifacts: ArtifactsType = None,
             dataset: Optional[dict] = None,
             explanations: Optional[str] = None,
             oodd: Optional[str] = None,
             num_gpus: Optional[int] = 0,
             expect_error_on_example_request=False
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
    @param expect_error_on_example_request: if not set to True,
    model validation will not pass if error is returned
    Note: Please refer to
    https://navio.craftworks.io/docs/guides/navio-models/model_creation/#3-test-model-serving
    for testing the model serving.
    @return: path to the .zip model file
    """

    path = process_path(path)
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
    _ModelValidator()(path,
                      expect_error_on_example_request)
    shutil.make_archive(path, 'zip', path)
    return Path(path + '.zip')
