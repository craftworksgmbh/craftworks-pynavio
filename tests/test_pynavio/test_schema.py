import unittest
from typing import Any, Dict, List

import pandas as pd
import pytest

from pynavio import make_example_request

DATA = {'x': 1, 'y': 2., 'z': 'str', 't': '2020-01-01 00:00:00'}

DATA_FRAME = pd.DataFrame([DATA])

SCHEMA = {
    'x': {
        'type': 'int',
        'nullable': False
    },
    'y': {
        'type': 'float',
        'nullable': False
    },
    'z': {
        'type': 'string',
        'nullable': False
    },
    't': {
        'type': 'timestamp',
        'nullable': False
    }
}


def _data(*keys) -> Dict[str, Any]:
    return {key: DATA[key] for key in keys}


def _schema(*keys: str) -> List[Dict[str, Any]]:
    return [{
        'name': key,
        'sampleData': DATA[key],
        **SCHEMA[key]
    } for key in keys]


@pytest.mark.parametrize(
    'args, expected',
    [  # yapf fix
        ({
            'target': 'z',
            'data': _data('x', 'y', 'z'),
        }, {
            'featureColumns': _schema('x', 'y'),
            'targetColumns': _schema('z')
        }),
        ({
            'target': 'z',
            'data': DATA_FRAME[['x', 'y', 'z']]
        }, {
            'featureColumns': _schema('x', 'y'),
            'targetColumns': _schema('z')
        }),
        ({
            'target': 'y',
            'data': _data('x', 'y', 'z'),
            'feature_columns': ['x']
        }, {
            'featureColumns': _schema('x'),
            'targetColumns': _schema('y')
        }),
        ({
            'target': 'z',
            'data': _data('x', 'y', 'z', 't'),
            'datetime_column': 't',
            'feature_columns': ['y', 'x']
        }, {
            'featureColumns': _schema('y', 'x'),
            'targetColumns': _schema('z'),
            'dateTimeColumn': _schema('t')[0]
        }),
        ({
            'target': 'x',
            'data': _data('x', 'y', 't'),
            'datetime_column': 't',
            'min_rows': 10
        }, {
            'featureColumns': _schema('y'),
            'targetColumns': _schema('x'),
            'dateTimeColumn': _schema('t')[0],
            'minimumNumberRows': 10
        })
    ])
def test_make_example_request(args: dict, expected: dict) -> None:
    result = make_example_request(**args)
    case = unittest.TestCase()
    case.maxDiff = None
    case.assertDictEqual(result, expected)


@pytest.mark.parametrize(
    'args',
    [  # yapf fix
        ({
            'target': 'z',
            'datetime_column': 'z',
            'data': _data('x', 'y', 'z'),
        }), ({
            'target': 'z',
            'data': pd.DataFrame(),
        }), ({
            'target': 'z',
            'data': list(),
        }), ({
            'target': 'x',
            'data': _data('x'),
        }), ({
            'target': 'x',
            'data': _data('x'),
            'feature_columns': ['x']
        }),
        ({
            'target': 'x',
            'datetime_column': 't',
            'data': _data('x', 't'),
            'feature_columns': ['x', 't']
        }), ({
            'target': 'x',
            'min_rows': 10,
            'data': _data('x', 't'),
        }),
        ({
            'target': 'x',
            'min_rows': 0,
            'datetime_column': 't',
            'data': _data('x', 't'),
        })
    ])
def test_make_example_request_assertions(args: dict) -> None:
    with pytest.raises(AssertionError):
        make_example_request(**args)
