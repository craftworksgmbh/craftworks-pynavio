import pandas as pd
from typing import Any, Dict, List, Optional, Union
from tempfile import TemporaryDirectory
from pathlib import Path
from pigar.parser import parse_imports

from .utils.common import (_get_path_as_str, _generate_default_to_ignore_dirs,
                           ExampleRequestType)

from .utils.infer_code_paths import (get_name_to_module_path_map,
                                     _get_code_path )

from .utils.infer_dependencies import (_generate_requirements_txt_file,
                                       read_requirements_txt)


def make_example_request(data: Union[Dict[str, Any], pd.DataFrame],
                         target: str,
                         datetime_column: Optional[str] = None,
                         feature_columns: Optional[List[str]] = None,
                         min_rows: Optional[int] = None) -> ExampleRequestType:
    """ Generates a request schema for a navio model from data
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


def infer_imported_code_path(
        path: Union[str, Path],
        root_path: Union[str, Path],
        to_ignore_paths: Optional[List[str]] = None) -> List[str]:
    """
    known edge cases and limitations:
     - Can result in duplicated copies in code_paths
     if the the imports are inconsistent,
     e.g. in one place from pynavio.utils.common import get_module_path
     and in other place from utils.common import get_module_path
     (with adding more paths to PYTHONPATH)
    @param path: path of the module/file from which to infer
     the imported code paths
    @param to_ignore_paths:  list of paths to ignore.
     - Ignores a directory named *venv* or
     containing *site-packages* by default
    @return: list of imported code paths
    """
    path = _get_path_as_str(path)
    root_path = _get_path_as_str(root_path)

    if to_ignore_paths is None:
        to_ignore_paths = []

    if not to_ignore_paths:
        to_ignore_paths = _generate_default_to_ignore_dirs(root_path)

    imported_modules, _ = parse_imports(
        path,
        ignores=[f'{to_ignore_path}' for to_ignore_path in to_ignore_paths])

    name_to_module = get_name_to_module_path_map(imported_modules, root_path,
                                                 to_ignore_paths)

    code_paths = [
        _get_code_path(module_name, path)
        for module_name, path in name_to_module.items()
        if _get_code_path(module_name, path)
    ]
    return list(set(code_paths))


def infer_external_dependencies(
        module_path: Union[str, Path],
        to_ignore_paths: List[str] = None) -> List[str]:
    """
    infers pip requirement strings.
    known edge cases and limitations:
     - in case of some libs, e.g. for pytorch, installing via pip is not
     recommended when using conda
    and would result in a broken conda env
     - it might add packages, that are not being used ( e.g. import
     statements under conditional operators, with false condition)
     - it might not be able to detect all the required dependencies,
     in which case the user could append/extend the list manually
    @param module_path:
    @param to_ignore_paths: list of paths to ignore.
     -Ignores a directory named *venv* or containing *site-packages* by
     default
    @return: list of inferred pip requirements, e.g.
    ['mlflow==1.15.0', 'scikit_learn == 0.24.1']
    """
    with TemporaryDirectory() as tmp_dir:
        requirements_txt_file = Path(tmp_dir) / 'requirements.txt'
        _generate_requirements_txt_file(requirements_txt_file, module_path,
                                        to_ignore_paths)
        requirements = read_requirements_txt(requirements_txt_file)
    return requirements
