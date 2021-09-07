from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from utils.common import make_example_request, prediction_call, to_mlflow

from .explainer_traits import TabularExplainerTraits

TARGET = 'target'


class Minimal(mlflow.pyfunc.PythonModel):
    _columns = ['x', 'y']

    @prediction_call
    def predict(self, context, model_input):
        return model_input[self._columns] \
            .astype(float) \
            .eval('x + y') \
            .pipe(lambda s: {'prediction': s.tolist()})


class MinimalPlotly(Minimal, TabularExplainerTraits):

    @prediction_call
    def predict(self, context, model_input):
        if not self.has_background(model_input):
            return super().predict(context, model_input)

        background = self.select_data(model_input, True)
        data = self.select_data(model_input, False)
        predictions = super().predict(context, data)

        result = {
            **predictions, 'explanation': [
                self.draw_plotly_explanation(row) for _, row in data.iterrows()
            ]
        }

        return result


def make_data(path: Path) -> None:
    data = pd.DataFrame(np.random.rand(10, 3),
                        columns=[TARGET, *Minimal._columns])
    data.to_csv(path, index=False)


example_request = make_example_request(
    {
        TARGET: 1.1 * len(Minimal._columns),
        **{column: 1.1 for column in Minimal._columns}
    },
    target=TARGET)


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None):

    with TemporaryDirectory() as tmp_dir:
        dataset = None
        if with_data:
            data_path = f'{tmp_dir}/minimal.csv'
            make_data(data_path)
            dataset = dict(name='minimal-data', path=data_path)

        model = Minimal()

        dependencies = [np, pd, './models', './utils']
        if explanations == 'plotly':
            import plotly
            dependencies.append(plotly)
            model = MinimalPlotly()

        to_mlflow(model,
                  example_request,
                  dataset=dataset,
                  path=path,
                  explanations=explanations,
                  dependencies=dependencies,
                  oodd='default' if with_oodd else 'disabled')
