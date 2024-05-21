from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import mlflow

import pynavio

from .minimal import example_request, make_data

TARGET = 'target'


class AdditionalFields(mlflow.pyfunc.PythonModel):

    @pynavio.prediction_call
    def predict(self, context, model_input):
        return {
            'prediction': [1.] * model_input.shape[0],
            'extraPerPrediction': [{
                'value': i
            } for i in range(model_input.shape[0], 0, -1)],
            'extra': {
                'int': 1,
                'float': 2.,
                'str': 'a',
                'list': [1, 2., 'a'],
                'dict': {
                    'a': 1,
                    'b': 2.,
                    'c': [1, 2, 3]
                }
            }
        }


def setup(path: Path, *args, **kwargs):
    with TemporaryDirectory() as tmp_dir:
        pynavio.mlflow.to_navio(AdditionalFields(),
                                example_request=example_request,
                                code_path=kwargs.get('code_path'),
                                path=path,
                                pip_packages=['mlflow==2.9.2', 'pynavio==0.2.4'])
