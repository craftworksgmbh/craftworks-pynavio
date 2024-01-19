import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd

import pynavio

TARGET = 'target'


class MinimalBoolean(mlflow.pyfunc.PythonModel):
    _float_cols = ['x', 'y']
    _bool_col = 'bool_col'

    @pynavio.prediction_call
    def predict(self, context, model_input):
        logger = logging.getLogger('gunicorn.error')
        logger.info("Data types:\n%s\n", model_input.dtypes)
        vals = model_input[self._float_cols] \
            .astype(float) \
            .eval('x + y')
        flags = model_input[self._bool_col] \
            .map(lambda val: str(val).lower() == 'true')
        vals.loc[flags] = vals.loc[flags]**2
        return {'prediction': vals.tolist()}


def make_data(path: Path) -> None:
    data = pd.DataFrame(np.random.rand(10, 3),
                        columns=[TARGET, *MinimalBoolean._float_cols])
    data[MinimalBoolean._bool_col] = data.iloc[:, 0] < .5
    data.to_csv(path, index=False)


example_request = pynavio.make_example_request(
    {
        TARGET: 4,
        **{
            **{column: 1 for column in MinimalBoolean._float_cols}, MinimalBoolean._bool_col: True
        }
    },
    target=TARGET)


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, Path]]] = None):

    with TemporaryDirectory() as tmp_dir:
        dataset = None
        if with_data:
            data_path = f'{tmp_dir}/minimal_bool.csv'
            make_data(data_path)
            dataset = dict(name='minimal-bool-data', path=data_path)

        pynavio.mlflow.to_navio(MinimalBoolean(),
                                example_request=example_request,
                                dataset=dataset,
                                path=path,
                                code_path=code_path,
                                explanations='disabled',
                                pip_packages=['mlflow==2.9.2', 'pynavio==0.2.4'],
                                oodd='default' if with_oodd else 'disabled')
