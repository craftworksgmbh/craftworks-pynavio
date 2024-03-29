import copy
from typing import Dict

PREDICTION_KEY = 'prediction'
not_sequence_prediction_types = ['boolean', 'integer', 'number', 'string']

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        PREDICTION_KEY: {
            "oneOf": [
                {
                    "type": 'array',
                    "minItems": 1,
                    "items": {
                        "type": "boolean"
                    },
                },
                {
                    "type": 'array',
                    "minItems": 1,
                    "items": {
                        "type": "number"
                    },
                },
                {
                    "type": 'array',
                    "minItems": 1,
                    "items": {
                        "type": "string"
                    },
                },
            ]
        },
    },
    "required": [PREDICTION_KEY],
}

METADATA_SCHEMA = {
    'type': 'object',
    'properties': {
        'request_schema': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string'
                }
            }
        },
        'dataset': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string'
                },
                'path': {
                    'type': 'string'
                }
            }
        },
        'oodDetection': {
            'type': 'string',
            'enum': ['disabled', 'default']
        },
        'explanations': {
            'type': 'string',
            'enum': ['disabled', 'default', 'plotly']
        },
        '.shm_mount': {
            'type': 'string'
        }
    },
    'required': ['request_schema']
}

COLUMN_SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {
            'type': 'string'
        },
        'sampleData': {
            'type': ['boolean', 'integer', 'number', 'array', 'string']
        },
        'type': {
            'type':
                'string',
            'enum': [
                'float', 'string', 'image', 'list', 'bool', 'int', 'timestamp'
            ]
        },
        'nullable': {
            'type': 'boolean'
        }
    },
    'required': ['name', 'sampleData', 'type', 'nullable']
}

REQUEST_SCHEMA_SCHEMA = {
    'type': 'object',
    'properties': {
        'featureColumns': {
            'minItems': 1,
            'type': 'array',
            'items': COLUMN_SCHEMA
        },
        'targetColumns': {
            'minItems': 1,
            'type': 'array',
            'items': COLUMN_SCHEMA
        },
        'dateTimeColumn': COLUMN_SCHEMA
    },
    'required': ['featureColumns', 'targetColumns']
}


def _not_nested_columns_schema() -> Dict:
    not_nested_col_schema = copy.deepcopy(COLUMN_SCHEMA)
    not_nested_col_schema['properties']['sampleData']['type'] = [
        'boolean', 'integer', 'number', 'string'
    ]
    not_nested_col_schema['properties']['type']['enum'] = [
        'float', 'string', 'image', 'bool', 'int', 'timestamp'
    ]
    return not_nested_col_schema


def not_nested_request_schema() -> Dict:
    not_nested_schema = copy.deepcopy(REQUEST_SCHEMA_SCHEMA)

    not_nested_schema['properties']['featureColumns']['items'] =\
        _not_nested_columns_schema()
    not_nested_schema['properties']['dateTimeColumn'] =\
        _not_nested_columns_schema()
    return not_nested_schema
