from tempfile import TemporaryDirectory
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from utils.common import make_example_request, prediction_call, to_mlflow

from .explainer_traits import TimeSeriesExplainerTraits

TARGET = 'activity'
DATETIME_COLUMN = 'Time'


class Timeseries(mlflow.pyfunc.PythonModel, TimeSeriesExplainerTraits):

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
                 background: pd.DataFrame) -> list:
        import shap

        explainer = shap.TreeExplainer(self._model)
        values = explainer.shap_values(data[self._column_order])

        df = pd.DataFrame(np.stack(values)[predictions,
                                           np.arange(predictions.shape[0]), :],
                          columns=self._column_order)

        # for each row of predictions, return a heatmap, where only columns
        # for that row have non-zero values. This is to indicate that
        # only data from one time point has an effect on the prediction.
        # Not using the datetime column for reindexing just in case there
        # are duplicated time values
        return [
            df.loc[[i]].reindex(np.arange(df.shape[0])).fillna(0.).set_index(
                data[DATETIME_COLUMN]) for i in range(data.shape[0])
        ]

    @prediction_call
    def predict(self, context, model_input):
        if (self._explanation_format in [None, 'default'] or
                not self.has_background(model_input)):
            return self._predict(model_input)

        background = self.select_data(model_input, True)
        data = self.select_data(model_input, False)

        predictions = self._model.predict(data[self._column_order])

        return {
            'prediction': [
                self._classes[prediction] for prediction in predictions
            ],
            'explanation': [
                self.draw_plotly_explanation(df)
                for df in self._explain(predictions, data, background)
            ]
        }


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None):

    data = pd.read_csv('data/activity_recognition.csv') \
        .fillna({TARGET: 'UNKNOWN'}) \
        .astype({TARGET: 'category'}) \
        .sort_values(by=[DATETIME_COLUMN, 'user', 'experiment']) \
        .drop(['user', 'experiment'], axis=1)

    column_order = data \
        .drop([TARGET, DATETIME_COLUMN], axis=1) \
        .columns \
        .tolist()

    example_request = make_example_request(
        data.to_dict(orient='records')[0], TARGET, DATETIME_COLUMN)

    model = RandomForestClassifier(verbose=1,
                                   n_jobs=4,
                                   n_estimators=33,
                                   max_depth=10)

    model.fit(data[column_order], data[TARGET].cat.codes)

    with TemporaryDirectory() as tmp_dir:
        dataset = None
        if with_data:
            data_path = f'{tmp_dir}/timeseries.csv'
            data.to_csv(data_path, index=False)
            dataset = dict(name='timeseries-data', path=data_path)

        mlflow_model = Timeseries(data[TARGET].cat.categories.tolist(),
                                  column_order, explanations)

        model_path = f'{tmp_dir}/model.joblib'
        joblib.dump(model, model_path)

        dependencies = [np, pd, sklearn, './models', './utils']
        if explanations == 'plotly':
            import plotly
            import shap
            dependencies.extend([plotly, shap])

        to_mlflow(mlflow_model,
                  example_request,
                  artifacts={'model': model_path},
                  dataset=dataset,
                  path=path,
                  explanations=explanations,
                  dependencies=dependencies,
                  oodd='default' if with_oodd else 'disabled')
