import logging
from tempfile import TemporaryDirectory
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from utils.common import make_example_request, prediction_call, to_mlflow

from .minimal import example_request

TARGET = 'target'


class ErrorModel(mlflow.pyfunc.PythonModel):

    @prediction_call
    def predict(self, context, model_input):
        raise NotImplementedError(
            'The predict call of this model deliberately throws an error!\n'
            f'Received data:\n {model_input.to_json(orient="records")}')


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None):

    with TemporaryDirectory() as tmp_dir:
        model = ErrorModel()
        dependencies = [np, pd, './models', './utils']
        to_mlflow(model, example_request, path=path, dependencies=dependencies)
