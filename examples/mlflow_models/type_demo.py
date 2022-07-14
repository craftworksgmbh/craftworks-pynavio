from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import mlflow
import numpy as np
import pandas as pd

import pynavio

TARGET = 'target'


class TypeDemo(mlflow.pyfunc.PythonModel):

    columns = ['int', 'float', 'str', 'bool', 'date', 'datetime']

    def _apply_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.astype({'int': int, 'float': float, 'bool': bool})
        for col in ['date', 'datetime']:
            frame[col] = pd.to_datetime(frame[col])
        return frame

    @pynavio.prediction_call
    def predict(self, context, model_input: pd.DataFrame) -> dict:
        frame = self._apply_schema(model_input)

        _sum = frame['int'] + frame['float']

        # square rows with even dates
        flag = (frame['date'].dt.day % 2) == 0
        _sum.loc[flag] = _sum.loc[flag]**2

        # scale rows where bool == true
        flag = frame['bool']
        _sum.loc[flag] = _sum.loc[flag] * 2

        # offset in rows where str == "yes"
        flag = frame['str'] == 'yes'
        _sum.loc[flag] = _sum.loc[flag] + .5

        # offset in rows where hour is even
        flag = (frame['datetime'].dt.hour % 2) == 0
        _sum.loc[flag] = _sum.loc[flag] + .3

        return {'prediction': _sum.tolist()}


def _make_data() -> pd.DataFrame:
    num_rows = 100
    return pd.DataFrame({
        'int': (np.random.rand(num_rows) * 10).astype(int),
        'float':
            np.random.rand(num_rows),
        'str':
            np.where(np.random.rand(num_rows) < .5, 'yes', 'no'),
        'bool':
            np.random.rand(num_rows) < .5,
        'date':
            pd.date_range(pd.Timestamp.now().round('d'),
                          pd.Timestamp.now() + pd.Timedelta('365d'),
                          freq='d').to_series().sample(num_rows).values,
        'datetime':
            pd.date_range(pd.Timestamp.now().round('h'),
                          pd.Timestamp.now() + pd.Timedelta('365d'),
                          freq='h').to_series().sample(num_rows).values,
        TARGET:
            np.zeros(num_rows)
    })


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, Path]]] = None):

    with TemporaryDirectory() as tmp_dir:
        data = _make_data()
        sample = data.head()  #.astype({'date': str, 'datetime': str})

        dataset = None
        if with_data:
            data_path = f'{tmp_dir}/data.csv'
            data.to_csv(data_path, index=False)
            dataset = dict(name='type-demo-data', path=data_path)

        pynavio.mlflow.to_navio(TypeDemo(),
                                example_request=pynavio.make_example_request(
                                    sample, TARGET),
                                explanations=explanations,
                                path=path,
                                pip_packages=['mlflow'],
                                code_path=code_path,
                                dataset=dataset,
                                oodd='default' if with_oodd else 'disabled')
