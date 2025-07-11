import copy
import json
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union

import jsonschema
import mlflow
import pandas as pd
import requests
import yaml

from pynavio.utils import ExampleRequestType, make_env
from pynavio.utils.json_encoder import JSONEncoder
from pynavio.utils.schemas import (METADATA_SCHEMA, PREDICTION_SCHEMA,
                                   REQUEST_SCHEMA_SCHEMA,
                                   not_nested_request_schema)

MODEL_SIZE_LIMIT_IN_BYTES = 1000_000_000
EXAMPLE_REQUEST = 'example_request'
ARTIFACTS = 'artifacts'
ArtifactsType = Optional[Dict[str, str]]
ERROR_KEYS = {'error_code', 'message', 'stack_trace'}
PREDICTION_KEY = 'prediction'
pynavio_model_validation = '(pynavio model validation)'

METADATA = 'metadata'
OOD_DETECTION = 'oodDetection'
MLMODEL = 'MLmodel'
REQUEST_SCHEMA = 'request_schema'
DATASET = 'dataset'
EXPLANATIONS = 'explanations'


def _is_ood_set_to_default_in_metadata(metadata: dict) -> bool:
    return metadata.get(OOD_DETECTION, '') == 'default'


def _is_explanation_set_to_default_in_metadata(metadata: dict) -> bool:
    return metadata.get(EXPLANATIONS, '') == 'default'


def _is_data_provided_in_metadata(metadata: dict) -> bool:
    return DATASET in metadata


def _is_default_ood_enabled_in_metadata(metadata: dict) -> bool:
    return _is_ood_set_to_default_in_metadata(metadata) and \
           _is_data_provided_in_metadata(metadata)


def _is_default_explanation_enabled_in_metadata(metadata: dict) -> bool:
    return _is_explanation_set_to_default_in_metadata(metadata) and \
           _is_data_provided_in_metadata(metadata)


def check_zip_size(model_zip, model_size_in_bytes):
    if Path(model_zip).stat().st_size > model_size_in_bytes:
        print(f"Warning: the default model.zip size limit is"
              f" {model_size_in_bytes} bytes. Please reduce the"
              f" size or contact craftworks support team to"
              f" increase the default size")


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
    assert any(item is not None for item in [example_request, artifacts]), \
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
                  num_gpus: Optional[int] = 0,
                  metadata: Optional[dict] = None) -> None:
    """
    The mlflow.pyfunc.save_model function supports the metadata argument for
    mlflow>=2.1.0 only.
    """
    # Path to MLmodel file
    path = Path(model_path) / 'MLmodel'

    # Read MLmodel file
    with path.open('r') as file:
        cfg = yaml.safe_load(file)

    # Init metadata
    cfg['metadata'] = metadata or {}

    # Add request_schema
    example_request_path_yml = 'flavors.python_function.artifacts.' \
                               'example_request.path'
    cfg['metadata'].update(
        request_schema=dict(path=_get_field(cfg, example_request_path_yml))
    )

    # Add dataset
    if dataset is not None:
        dataset_path_yml = 'flavors.python_function.artifacts.dataset.path'
        cfg['metadata'].update(dataset=dataset)
        cfg['metadata']['dataset']['path'] = _get_field(cfg, dataset_path_yml)

    # Add explanations
    explanations = explanations or 'default'
    accepted_values = ['disabled', 'default', 'plotly']
    assert explanations in accepted_values, \
        f'explanations config must be one of {accepted_values}'
    cfg['metadata'].update(explanations=explanations)

    # Add oodDetection
    oodd = oodd or 'default'
    accepted_values = ['disabled', 'default']
    assert oodd in accepted_values, \
        f'oodd config must be one of {accepted_values}'
    cfg['metadata'].update(oodDetection=oodd)

    # Add gpus
    assert num_gpus >= 0, 'num_gpus cannot be negative'
    if num_gpus > 0:
        cfg['metadata'].update(gpus=num_gpus)

    # Write MLmodel file
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
        'dataset': pd.read_csv(data_path) if data_path is not None else None,
        'explanation_format': _get_field(yml, 'metadata.explanations.format'),
        'example_request': example_request
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


def _validate_schema(data, schema, name='', raise_exception=True):
    """
    Validate the given data against the specified JSON schema.

    @param data: The data to validate.
    @param schema: The jsonschema schema to validate against.
    @param name: A descriptive name for the data being validated,
        used in error message
    @param raise_exception: Whether to raise an exception if the
        validation fails. If True (default), the function raises a
        jsonschema ValidationError exception with a
        descriptive error message.
    @return: True if the validation passes, False otherwise.
     Only returned if raise_exception is False.
    """
    is_valid = True
    try:
        jsonschema.validate(data, schema)
    except jsonschema.exceptions.ValidationError:
        if raise_exception:
            print(f"Error: {pynavio_model_validation} "
                  f"Error during {name} validation")
            raise
        else:
            is_valid = False
    return is_valid


def is_input_nested(example_request, not_nested_schema):
    is_not_nested = _validate_schema(example_request,
                                     not_nested_schema,
                                     '',
                                     raise_exception=False)
    return not is_not_nested


def _is_wrapped_by_prediction_call(func):
    return getattr(func, '__wrapped_by_prediction_call__', False)


def _is_model_predict_wrapped_by_prediction_call(model):
    return _is_wrapped_by_prediction_call(
        model._model_impl.python_model.predict)


class ModelValidator:
    """
    A utility class for validating navio mlflow models.
    Raises jsonschema.exceptions.ValidationError and AssertionError if
    there are errors in validation.
    Example usage:
        >>> ModelValidator()(model_path = '/path/to/my/model')

    """

    @staticmethod
    def validate_metadata(model_path):
        """
        Validate the metadata: example request file and MLmodel file
        of the given navio mlflow model.

        @param model_path: The directory path of the model to validate.
        @raises jsonschema.exceptions.ValidationError:
        If there is an error during the validation process.
        @return: None
        """
        config = _read_mlmodel_yaml(model_path)
        metadata = config.get('metadata')
        _validate_schema(metadata, METADATA_SCHEMA, "MLmodel")

        example_request = _read_example_request(model_path, config)
        _validate_schema(example_request, REQUEST_SCHEMA_SCHEMA,
                         "example request")
        if is_input_nested(example_request, not_nested_request_schema()):
            print('Warning: {pynavio_model_validation} the nested'
                  ' model input is not supported'
                  ' by frontend rendering, it will only be possible'
                  ' to see the example request as plain json in the'
                  ' try-out or deployment views. Consider using'
                  ' string representation of nested example in the'
                  ' example request json.')
            if _is_default_ood_enabled_in_metadata(metadata) or \
                    _is_default_explanation_enabled_in_metadata(metadata):
                print("Warning: {pynavio_model_validation} default"
                      " ood and explanations"
                      " are not supported for nested model inputs.")

    @staticmethod
    def run_model_io(model_path, model_input=None, **kwargs):
        """
        Run the given navio mlflow model file with the given input.

        @param model_path: The directory path of the model to run.
        @param model_input: optional, the input data for the model's
        predict method. If None, the example request specified in
        the model metadata will be used as input.
        @param kwargs: Additional keyword arguments.

        @return: The input data and the prediction output.
        """
        model = mlflow.pyfunc.load_model(model_path)
        if model_input is None:
            model_input = _get_example_request_df(model_path)
        return model_input, model.predict(model_input)

    @staticmethod
    def _check_if_prediction_call_is_used(model_path):
        model = mlflow.pyfunc.load_model(model_path)
        used = True  # do not print warning if not sure
        # try checking the original model's predict function
        try:
            used = _is_model_predict_wrapped_by_prediction_call(model)
        except AttributeError:
            pass
        if not used:
            print(f"Warning: {pynavio_model_validation} Please use"
                  f" pynavio.prediction_call to decorate"
                  f" the predict method of the model, which will add the"
                  f" needed error keys({ERROR_KEYS}) for error"
                  f" case to see descriptive"
                  f" errors from navio for ease of debugging.")

    @staticmethod
    def check_zip_size(model_zip, model_size_in_bytes):
        if Path(model_zip).stat().st_size > model_size_in_bytes:
            print(f"Warning: {pynavio_model_validation} "
                  f"the default model.zip size limit is"
                  f" {model_size_in_bytes} bytes. Please reduce the"
                  f" size or contact craftworks support team to"
                  f" increase the default size")

    @staticmethod
    def verify_model_output(model_output, **kwargs):
        """
        Verify the output of the navio mlflow model.

        @param model_output: The output of the model.
        @param kwargs: Additional keyword arguments.

        @raises AssertionError: If the output is not valid.
        @return: None
        """

        def _validate_prediction_schema(model_prediction):
            try:
                jsonschema.validate(model_prediction, PREDICTION_SCHEMA)
            except jsonschema.exceptions.ValidationError:
                print(f"ERROR: {pynavio_model_validation} The value of "
                      f"model_output['{PREDICTION_KEY}']"
                      " must be one of the following types "
                      "(cannot be nested or mixed type): "
                      "'array','boolean', 'integer', 'number', 'string'")
                raise

        assert isinstance(model_output, Mapping), "Model " \
            f"ERROR: {pynavio_model_validation} output has to be a dictionary"

        if PREDICTION_KEY in model_output:
            _validate_prediction_schema(model_output)
        else:
            assert set(model_output.keys()) == ERROR_KEYS, \
                f"ERROR: {pynavio_model_validation}" \
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

    def _run(self, model_path, model_zip, model_zip_size_limit, **kwargs):
        self.validate_metadata(model_path)
        model_input, model_output = self.run_model_io(model_path)
        self._check_if_prediction_call_is_used(model_path)
        self.verify_model_output(model_output)
        self.check_zip_size(model_zip, model_zip_size_limit)

    def __call__(self, model_path, model_zip, model_zip_size_limit, **kwargs):
        try:
            self._run(model_path, model_zip, model_zip_size_limit, **kwargs)
        except (jsonschema.exceptions.ValidationError, AssertionError):
            print(f'{pynavio_model_validation}: Validation failed. Please fix'
                  f' the identified issues before uploading the model.'
                  f'{kwargs.get("append_to_failed_msg", "")}')
            raise
        print(f'{pynavio_model_validation}: Validation succeeded.'
              f'{kwargs.get("append_to_succeeded_msg", "")}')


def _is_mlflow2():
    import mlflow
    from packaging import version
    return version.parse(mlflow.__version__) >= version.parse("2.0.0")


def _convert_to_mlflow2_format(request_data):
    dataframe_records = pd.DataFrame.from_records(
        columns=request_data['columns'],
        data=request_data['data']).\
        to_json(orient='records')
    request_data = {"dataframe_records": json.loads(dataframe_records)}
    return request_data


def check_model_serving(model_path: Union[str, Path],
                        port=5001,
                        request_bodies=None):
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
    time.sleep(10)
    response = None

    try:
        for data in (request_bodies or _fetch_data(model_path)):
            if _is_mlflow2():
                data = _convert_to_mlflow2_format(data)
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


def _is_valid_sys_dependency_list(sys_dependencies: List[str]) -> bool:
    return isinstance(sys_dependencies, List) and \
        all(isinstance(item, str) for item in sys_dependencies)


def _add_sys_dependencies(path: str, sys_dependencies: List[str]) -> None:
    """
    Writes the system dependencies to a text file, one per line.

    :param path: Path to save the sys_dependencies.txt file.
    :param sys_dependencies: List of system dependencies.
    :return: None
    """
    if sys_dependencies is None:
        return

    assert _is_valid_sys_dependency_list(sys_dependencies)

    file_path = Path(path) / 'sys_dependencies.txt'

    with open(file_path, 'w') as f:
        f.write("\n".join(sys_dependencies))


def to_navio(model: mlflow.pyfunc.PythonModel,
             path,
             example_request: ExampleRequestType = None,
             pip_packages: List[str] = None,
             extra_pip_packages: List[str] = None,
             code_path: Optional[List[Union[str, Path]]] = None,
             conda_packages: List[str] = None,
             sys_dependencies: List[str] = None,
             conda_channels: List[str] = None,
             conda_env: str = None,
             artifacts: ArtifactsType = None,
             dataset: Optional[dict] = None,
             explanations: Optional[str] = None,
             oodd: Optional[str] = None,
             num_gpus: Optional[int] = 0,
             validate_model: Optional[bool] = True,
             metadata: Optional[dict] = None) -> Path:
    """
    create a .zip mlflow model file for navio
    Usage: If both pip_packages or conda_env are not set, then
    the env will be inferred by mlflow. If there are some specific
    packages needed the argument extra_pip_packages can be used. The
    arguments pip_packages, extra_pip_packages and conda_env are
    mutually exclusive, only one can be set.

    @param model: model to save
    @param path: path of where model .zip file needs to be saved
    @param example_request: example_request for the given model.
    If not set, needs to be present in artifacts.
    @param pip_packages: list of pip packages(optionally with versions)
    with the syntax of a requirements.txt file, e.g.
    ['mlflow==1.15.0', 'scikit_learn == 0.24.1'].
    Tip: For most cases it should be enough to use
    pynavio.utils.infer_dependencies.infer_external_dependencies().
    @param extra_pip_packages: list of extra necessary
    pip packages (optionally with versions). This option is to be used in
    case if the env is inferred by mlflow and there is a need to add
    extra pip packages (pip_packages or conda_env are not set).
    @param code_path: A list of local filesystem paths to Python file
    dependencies (or directories containing file dependencies)
    @param conda_packages: list of conda packages
    @param sys_dependencies: list of system library dependencies
    @param conda_channels: list of conda channels
    @param conda_env: the path of a conda.yaml file to use. If specified,
    the values of conda_channels, conda_packages and pip_packages would be
    ignored.
    @param artifacts: If not set, need to set example_request
    @param dataset:
    @param explanations: expected values are ['disabled', 'default', 'plotly']
     If not set, 'default' is used
    @param oodd: expected values are ['disabled', 'default'].
     If not set, 'default' is used
    @param num_gpus:
    @param validate_model: if the output model should be validated by
     ModelValidator. On by default(True), to disable set to False.
    @param metadata: metadata dictionary to be added to the metadata section
     in the MLmodel file

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

        if any(
                Path(code_p).resolve() in Path(path).resolve().parents
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

        assert sum(x is not None for x in [extra_pip_packages, pip_packages,
                                           conda_env]) <= 1, \
            "The arguments 'extra_pip_packages', 'pip_packages' " \
            "and 'conda_env' cannot be specified at the same time."

        conda_env = make_env(pip_packages, conda_packages, conda_channels,
                             conda_env)

        code_path = _safe_code_path(code_path)

        artifacts = register_example_request(tmp_dir, example_request,
                                             artifacts)

        shutil.rmtree(path, ignore_errors=True)

        mlflow.pyfunc.save_model(path=path,
                                 python_model=model,
                                 conda_env=conda_env,
                                 extra_pip_requirements=extra_pip_packages,
                                 artifacts=artifacts,
                                 code_path=code_path)

        _add_metadata(path,
                      dataset=dataset,
                      explanations=explanations,
                      oodd=oodd,
                      num_gpus=num_gpus,
                      metadata=metadata)
        _add_sys_dependencies(path, sys_dependencies)
        shutil.make_archive(path, 'zip', path)
        model_zip = Path(path + '.zip')

    if validate_model:
        msg_kwargs = {
            'append_to_failed_msg': ' To disable validation set '
                                    'validate_model to False',
            'append_to_succeeded_msg': ' Note: Please refer to '
                                       'check_model_serving()'
                                       ' method and '
                                       'https://navio.craftworks.io/'
                                       'docs/guides/navio-models/'
                                       'model_creation/'
                                       '#3-test-model-serving'
                                       ' for testing the model '
                                       'serving.',
        }
        ModelValidator()(path, model_zip, MODEL_SIZE_LIMIT_IN_BYTES,
                         **msg_kwargs)

    return model_zip
