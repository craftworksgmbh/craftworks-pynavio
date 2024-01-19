from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import mlflow
import numpy as np
import pandas as pd

import pynavio

TARGET = 'target'


class Minimal(mlflow.pyfunc.PythonModel):
    _columns = ['x', 'y']

    @pynavio.prediction_call
    def predict(self, context, model_input):
        return model_input[self._columns] \
            .astype(float) \
            .eval('x + y') \
            .pipe(lambda s: {'prediction': s.tolist()})


class MinimalPlotly(Minimal, pynavio.traits.TabularExplainerTraits):

    def __init__(self):
        super().__init__(explanation_format='plotly')

    @pynavio.prediction_call
    def predict(self, context, model_input):
        if not self.should_explain(model_input):
            return super().predict(context, model_input)

        background = self.select_data(model_input, True)
        data = self.select_data(model_input, False)
        predictions = super().predict(context, data)

        result = {
            **predictions, 'explanation': [
                self.draw_plotly_explanation(row)
                for _, row in data.astype(float).iterrows()
            ]
        }

        return result


def make_data(path: Path) -> None:
    data = pd.DataFrame(np.random.rand(10, 3),
                        columns=[TARGET, *Minimal._columns])
    data.to_csv(path, index=False)


example_request = pynavio.make_example_request(
    {
        TARGET: float(sum(range(len(Minimal._columns)))),
        **{col: float(i) for i, col in enumerate(Minimal._columns)}
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
            data_path = f'{tmp_dir}/minimal.csv'
            make_data(data_path)
            dataset = dict(name='minimal-data', path=data_path)

        model = Minimal()
        pip_packages = ['mlflow==2.9.2', 'pynavio==0.2.4']
        if explanations == 'plotly':
            model = MinimalPlotly()
            pip_packages.append('plotly')

        pynavio.mlflow.to_navio(model,
                                example_request=example_request,
                                dataset=dataset,
                                path=path,
                                code_path=code_path,
                                explanations=explanations,
                                pip_packages=pip_packages,
                                oodd='default' if with_oodd else 'disabled')
