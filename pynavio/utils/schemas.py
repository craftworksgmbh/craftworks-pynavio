
PREDICTION_KEY = 'prediction'
not_sequence_prediction_types = ['boolean', 'integer',
                                 'number', 'string']

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        PREDICTION_KEY: {"oneOf":
                         [
                          {"type": not_sequence_prediction_types},
                          {"type": 'array',
                           "minItems": 1,
                           "items": {"type": "boolean"},
                           },
                          {"type": 'array',
                           "minItems": 1,
                           "items": {"type": "number"},
                           },
                          {"type": 'array',
                           "minItems": 1,
                           "items": {"type": "string"},
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
                'path': {'type': 'string'}
            }
        },
        'dataset': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'path': {'type': 'string'}
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
        '.shm_mount': {'type': 'string'}
    },
    'required': ['request_schema']
}

COLUMN_SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'sampleData': {'type': [
            'boolean', 'integer', 'number', 'array', 'string'
        ]},
        'type': {
            'type': 'string',
            'enum': ['float', 'string', 'image', 'list', 'bool', 'int',
                     'timestamp']
        },
        'nullable': {'type': 'boolean'}
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
