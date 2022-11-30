from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import mlflow

import pynavio

from .minimal import example_request

TARGET = 'target'


class ErrorModel(mlflow.pyfunc.PythonModel):

    @pynavio.prediction_call
    def predict(self, context, model_input):
        raise NotImplementedError(
            'The predict call of this model deliberately throws an error!\n'
            f'Received data:\n {model_input.to_json(orient="records")}')


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, Path]]] = None,
          ):

    with TemporaryDirectory() as tmp_dir:
        pynavio.mlflow.to_navio(ErrorModel(),
                                example_request=example_request,
                                path=path,
                                code_path=code_path,
                                pip_packages=['mlflow']
                                )
