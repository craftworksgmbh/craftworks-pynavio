import inspect
import json
import logging
import platform
import shutil
import traceback
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

import mlflow
import pandas as pd
import pip
import yaml


def get_module_path(module: ModuleType) -> str:
    """ Use for local (non pip installed) modules only.
    This is useful for trainer models.
    """
    return str(Path(inspect.getfile(module)).parent)


def _make_conda_env(
    pip_packages: List[str] = None,
    conda_packages: List[str] = None,
    conda_channels: List[str] = None,
    conda_env: str = None,
) -> Dict[str, Any]:
    """
    makes the value for the mlflow.pyfunc.save_model()'s conda_env argument
    Usage: either pip_packages or conda_env need to be set.
    @param pip_packages:
    @param conda_packages:
    @param conda_channels:
    @param conda_env: the path of a conda.yaml file to use. If specified,
    the values of conda_channels, conda_packages and pip_packages would be
    ignored.
    @return:
    """
    assert any(item is not None for item in [pip_packages, conda_env]),\
        "either 'pip_packages' or 'conda_env' need to be set"

    if conda_env is None:
        conda_env = {
            'channels': ['defaults', 'conda-forge', *(conda_channels or [])],
            'dependencies': [
                f'python={platform.python_version()}',
                f'pip={pip.__version__}', *(conda_packages or []), {
                    'pip': pip_packages
                }
            ],
            'name': 'venv'
        }

    return conda_env


def _make_artifact(tmp_dir, example_request, artifacts):
    artifacts = {
        'example_request': f'{tmp_dir}/example_request.json',
        **(artifacts or {})
    }
    with open(artifacts['example_request'], 'w') as file:
        json.dump(example_request, file, indent=4)
    return artifacts


def _safe_code_path(code_path):
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

    cfg.update(metadata=dict(request_schema=dict(
        path='artifacts/example_request.json')))

    if dataset is not None:
        cfg['metadata'].update(dataset=dataset)

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


def to_navio_mlflow(model: mlflow.pyfunc.PythonModel,
                    example_request: ExampleRequest,
                    path: Union[str, Path],
                    pip_packages: List[str] = None,
                    code_path: List = None,
                    conda_packages: List[str] = None,
                    conda_channels: List[str] = None,
                    conda_env: str = None,
                    artifacts: Optional[Dict[str, str]] = None,
                    dataset: Optional[dict] = None,
                    explanations: Optional[str] = None,
                    oodd: Optional[str] = None,
                    num_gpus: Optional[int] = 0) -> Path:
    """
    create a .zip mlflow model file for navio
    Usage: either pip_packages or conda_env need to be set.

    @param model: model to save
    @param example_request: example_request for the given model
    @param path: path of where model .zip file needs to be saved
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
    @param artifacts:
    @param dataset:
    @param explanations:
    @param oodd:
    @param num_gpus:
    @return: path to the .zip model file
    """

    path = str(path)

    with TemporaryDirectory() as tmp_dir:
        if dataset is not None:
            _check_data_spec(dataset)
            artifacts = artifacts or dict()
            artifacts.update(dataset=dataset['path'])
            dataset.update(path=f'artifacts/{Path(dataset["path"]).parts[-1]}')

        conda_env = _make_conda_env(pip_packages, conda_packages,
                                    conda_channels, conda_env)

        code_path = _safe_code_path(code_path)

        artifacts = _make_artifact(tmp_dir, example_request, artifacts)

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

    model = mlflow.pyfunc.load_model(path)  # test load
    shutil.make_archive(path, 'zip', path)
    return Path(path + '.zip')


def make_example_request(data: Union[Dict[str, Any], pd.DataFrame],
                         target: str,
                         datetime_column: Optional[str] = None,
                         feature_columns: Optional[List[str]] = None,
                         min_rows: Optional[int] = None) -> ExampleRequest:
    """ Generates a request schema for a navio MLflow model from data
    @param data: a sample of data to use for schema generation
    @param target: name of the target column
    @param datetime_column: name of the datetime column (for time series)
    @param feature_columns: list of names of feature columns
    @param min_rows: minimal number of rows of data the model expects
    """

    assert target != datetime_column, \
        'Target column name must not be equal to that of datetime column'

    type_names = {int: 'float', float: 'float', str: 'string'}

    def _column_spec(name: str, _type: Optional[str] = None) -> Dict[str, Any]:
        return {
            "name": name,
            "sampleData": row[name],
            "type": _type or type_names[type(row[name])],
            "nullable": False
        }

    if isinstance(data, pd.DataFrame):
        assert data.shape[0] > 0, \
            'Cannot generate example request from an empty data frame'
        row = data.iloc[:1].to_dict(orient='records')[0]
    else:
        assert isinstance(data, dict), \
            f'Expected data to be of type dict, got type {type(data)}'
        row = data

    if feature_columns is None:
        feature_columns = set(row.keys()) - {target, datetime_column}
        feature_columns = sorted(list(feature_columns))  # to ease testing
        assert feature_columns, \
            'Could not infer a valid set of feature columns based on the ' \
            f'data columns: {list(row.keys())}, with target={target} and ' \
            f'datetime_column={datetime_column}'

    assert feature_columns, 'Empty feature column list is not allowed'

    assert target not in feature_columns, \
        f'Feature column list must not contain target column {target}'

    assert datetime_column not in feature_columns, \
        'Feature column list must not contain datetime '\
        f'column {datetime_column}'

    example = {
        "featureColumns": [_column_spec(name) for name in feature_columns],
        "targetColumns": [_column_spec(target)]
    }

    if datetime_column is None:
        assert min_rows is None, \
            'min_rows is only allowed when a datetime column is specified'
        return example

    example['dateTimeColumn'] = _column_spec(datetime_column, 'timestamp')
    if min_rows is None:
        return example

    assert min_rows > 0, f'Expected min_rows > 0, got min_rows = {min_rows}'

    example['minimumNumberRows'] = min_rows
    return example


def prediction_call(predict_fn: callable) -> callable:
    logger = logging.getLogger('gunicorn.error')

    @wraps(predict_fn)
    def wrapper(*args, **kwargs) -> dict:
        try:
            return predict_fn(*args, **kwargs)
        except Exception as exc:
            logger.exception('Prediction call failed')
            return {
                'error_code': exc.__class__.__name__,
                'message': str(exc),
                'stack_trace': traceback.format_exc()
            }

    return wrapper
