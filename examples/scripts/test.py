import copy
import json
import glob
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yaml

from pynavio.mlflow import _get_field

URL = 'http://127.0.0.1:5001/invocations'


def _read_metadata(model_path: str) -> dict:
    with (Path(model_path) / 'MLmodel').open('r') as file:
        yml = yaml.safe_load(file)

    data_path = _get_field(yml, 'metadata.dataset.path')
    data_path = Path(model_path) / data_path if data_path is not None else None
    example_request_path = Path(model_path) / _get_field(yml, 'metadata.request_schema.path')
    with open(example_request_path, 'r') as file:
        example_request = json.load(file)

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


def mlflow_to_navio(data: dict) -> dict:
    return {'rows': [dict(zip(data['columns'], row)) for row in data['data']]}


def _check_model_serving(model_path):
    process = subprocess.Popen(
        f'mlflow models serve -m {model_path} -p 5001 --no-conda'.split())

    time.sleep(5)

    try:
        for data in _fetch_data(model_path):
            response = requests.post(URL,
                                     data=json.dumps(data, allow_nan=True),
                                     headers={'Content-type':
                                              'application/json'})
            response.raise_for_status()
    finally:
        process.terminate()
        print(response.json())
        subprocess.run('pkill -f gunicorn'.split())
        time.sleep(2)


if __name__ == '__main__':
    model_path = sys.argv[1]
    _check_model_serving(model_path)
