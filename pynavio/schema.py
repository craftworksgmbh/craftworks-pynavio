from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .utils.common import ExampleRequestType


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

    type_names = {
        int: 'int',
        float: 'float',
        str: 'string',
        bool: 'string',
        pd.Timestamp: 'timestamp'
    }

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
