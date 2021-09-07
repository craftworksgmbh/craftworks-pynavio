from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from utils.common import make_example_request, prediction_call, to_mlflow

from .explainer_traits import TabularExplainerTraits
from .minimal import example_request, make_data

TARGET = 'target'


class AdditionalFields(mlflow.pyfunc.PythonModel):

    @prediction_call
    def predict(self, context, model_input):
        return {
            'prediction': [*range(model_input.shape[0])],
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
        dependencies = [np, pd, './models', './utils']
        to_mlflow(AdditionalFields(),
                  example_request,
                  path=path,
                  dependencies=dependencies)
