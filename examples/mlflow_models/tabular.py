from pathlib import PosixPath
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import pynavio

TARGET = 'target'


class Tabular(mlflow.pyfunc.PythonModel,
              pynavio.traits.TabularExplainerTraits):
    BG_COLUMN = 'is_background'

    def __init__(self, classes, column_order, explanation_format=None):
        self._classes = classes
        self._column_order = column_order
        self._explanation_format = explanation_format

    def load_context(self, context) -> None:
        self._model = joblib.load(context.artifacts['model'])

    def _predict(self, model_input) -> list:
        return pd.Series(
                self._model.predict(model_input[self._column_order])) \
            .map(self._classes.__getitem__) \
            .pipe(lambda s: {'prediction': s.tolist()})

    def _explain(self, predictions: np.ndarray, data: pd.DataFrame,
                 background: pd.DataFrame) -> pd.DataFrame:
        import shap

        explainer = shap.TreeExplainer(self._model)
        values = explainer.shap_values(data[self._column_order])

        return pd.DataFrame(
            np.stack(values)[predictions,
                             np.arange(predictions.shape[0]), :],
            columns=self._column_order)

    @pynavio.prediction_call
    def predict(self, context, model_input):
        if (self._explanation_format in [None, 'default'] or
                not self.has_background(model_input)):
            return self._predict(model_input)

        background = self.select_data(model_input, True)
        data = self.select_data(model_input, False)

        predictions = self._model.predict(data[self._column_order])
        explanations = self._explain(predictions, data, background)

        return {
            'prediction': [
                self._classes[prediction] for prediction in predictions
            ],
            'explanation': [
                self.draw_plotly_explanation(row)
                for _, row in explanations.iterrows()
            ]
        }


def load_data() -> pd.DataFrame:
    data = load_iris(as_frame=True)
    df = data['data'].rename(columns=lambda c: '_'.join(c.split()[:-1]))
    df[TARGET] = data['target_names'][data['target']]
    return df.astype({TARGET: 'category'})


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, PosixPath]]] = None):
    data = load_data()
    column_order = data.drop(TARGET, axis=1).columns.tolist()

    example_request = pynavio.make_example_request(
        data.to_dict(orient='records')[0], TARGET)

    model = RandomForestClassifier()
    model.fit(data[column_order], data[TARGET].cat.codes)

    with TemporaryDirectory() as tmp_dir:
        model_path = f'{tmp_dir}/model.joblib'
        joblib.dump(model, model_path)

        dataset = None
        if with_data:
            data_path = f'{tmp_dir}/iris.csv'
            data.to_csv(data_path, index=False)
            dataset = dict(name='tabular-data', path=data_path)

        pip_packages = ['mlflow==2.9.1', 'scikit-learn==1.2.2', 'joblib==1.3.2', 'pynavio==0.3.1']

        if explanations == 'plotly':
            pip_packages.extend(['plotly', 'shap'])

        pynavio.mlflow.to_navio(Tabular(data[TARGET].cat.categories.tolist(),
                                        column_order, explanations),
                                example_request=example_request,
                                explanations=explanations,
                                artifacts={'model': model_path},
                                path=path,
                                pip_packages=pip_packages,
                                code_path=code_path,
                                dataset=dataset,
                                oodd='default' if with_oodd else 'disabled')
