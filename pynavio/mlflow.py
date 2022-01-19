import json
import shutil
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union
import mlflow
import yaml

from pynavio.utils import ExampleRequestType, make_env

EXAMPLE_REQUEST = 'example_request'
ARTIFACTS = 'artifacts'
ArtifactsType = Optional[Dict[str, str]]


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
            json.dump(example_request, file, indent=4)
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


def to_navio(model: mlflow.pyfunc.PythonModel,
             path: Union[str, Path],
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
             num_gpus: Optional[int] = 0) -> Path:
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
    @return: path to the .zip model file
    """

    path = str(path)

    with TemporaryDirectory() as tmp_dir:
        if dataset is not None:
            _check_data_spec(dataset)
            artifacts = artifacts or dict()
            artifacts.update(dataset=dataset['path'])
            dataset.update(path=f'artifacts/{Path(dataset["path"]).parts[-1]}')

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

    model = mlflow.pyfunc.load_model(path)  # test load
    shutil.make_archive(path, 'zip', path)
    return Path(path + '.zip')
