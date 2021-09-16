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
import pip
import yaml


def get_module_path(module: ModuleType) -> str:
    """ Use for local (non pip installed) modules only.
    This is useful for trainer models.
    """
    return str(Path(inspect.getfile(module)).parent)


def _organize_dependencies(deps: List[Any]) -> tuple:
    """ Dependencies can be module objects or paths
    """
    assert all(isinstance(d, (str, Path, ModuleType)) for d in deps), \
        'Dependencies can only be modules or code paths'

    code, modules = ([*filter(lambda d: isinstance(d, (str, Path)), deps)],
                     [*filter(lambda d: isinstance(d, ModuleType), deps)])

    code_paths = list(map(Path, code))

    assert all(p.is_dir() for p in code_paths), \
        'All code dependencies must be directories'
    assert not any(p.resolve().absolute() == Path.cwd().absolute()
                   for p in code_paths), \
        'Code paths must not contain the current directory'

    # deleting __pycache__, otherwise MLFlow adds it to the code directory
    for path in code_paths:
        for cache_dir in path.glob('**/__pycache__'):
            shutil.rmtree(cache_dir, ignore_errors=True)

    return code, modules


def _make_mlflow_config(
        tmp_dir: str,
        dependencies: list,
        conda_packages: List[str] = None,
        artifacts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:

    def _module_name(module: ModuleType) -> str:
        return {
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow'
        }.get(module.__name__, module.__name__)

    code, modules = _organize_dependencies(dependencies)
    return {
        'artifacts': {
            'example_request': f'{tmp_dir}/example_request.json',
            **(artifacts or {})
        },
        'conda_env': {
            'channels': ['defaults', 'anaconda', 'pytorch'],
            'dependencies': [
                f'python={platform.python_version()}',
                f'pip={pip.__version__}', *(conda_packages or []), {
                    'pip': [
                        f'{_module_name(module)}=={module.__version__}'
                        for module in [mlflow, *modules]
                    ]
                }
            ],
            'name': 'venv'
        },
        'code_path': code if code else None
    }


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


def to_mlflow(model: mlflow.pyfunc.PythonModel,
              example_request: ExampleRequest,
              dependencies: List[Any],
              path: Union[str, Path],
              conda_packages: List[str] = None,
              artifacts: Optional[Dict[str, str]] = None,
              dataset: Optional[dict] = None,
              explanations: Optional[str] = None,
              oodd: Optional[str] = None,
              num_gpus: Optional[int] = 0) -> Path:

    path = str(path)

    with TemporaryDirectory() as tmp_dir:
        if dataset is not None:
            _check_data_spec(dataset)
            artifacts = artifacts or dict()
            artifacts.update(dataset=dataset['path'])
            dataset.update(path=f'artifacts/{Path(dataset["path"]).parts[-1]}')

        config = _make_mlflow_config(tmp_dir, dependencies, conda_packages,
                                     artifacts)

        with open(config['artifacts']['example_request'], 'w') as file:
            json.dump(example_request, file, indent=4)

        shutil.rmtree(path, ignore_errors=True)
        mlflow.pyfunc.save_model(path=path, python_model=model, **config)
        _add_metadata(path,
                      dataset=dataset,
                      explanations=explanations,
                      oodd=oodd,
                      num_gpus=num_gpus)

    model = mlflow.pyfunc.load_model(path)  # test load
    shutil.make_archive(path, 'zip', path)
    return Path(path + '.zip')


def make_example_request(row: Dict[str, Any],
                         target: str,
                         datetime_column: Optional[str] = None,
                         min_rows: Optional[int] = None) -> ExampleRequest:

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

    example = {
        "featureColumns": [
            _column_spec(name)
            for name in row.keys()
            if name != target and name != datetime_column
        ],
        "targetColumns": [_column_spec(target)]
    }

    if datetime_column is None:
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
