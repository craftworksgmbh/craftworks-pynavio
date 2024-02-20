import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

import pynavio

TARGET = 'activity'
DATETIME_COLUMN = 'Time'
GROUP_COLUMNS = ['user', 'experiment']


class Timeseries(mlflow.pyfunc.PythonModel,
                 pynavio.traits.TimeSeriesExplainerTraits):

    def __init__(self, classes, column_order, explanation_format=None):
        super().__init__(explanation_format=explanation_format)
        self._classes = classes
        self._column_order = column_order

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

    @pynavio.prediction_call
    def predict(self, context, model_input):
        if not self.should_explain(model_input):
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


def prepare_data(path: str) -> pd.DataFrame:
    logger = logging.getLogger('gunicorn.error')

    if path.lower().endswith('parquet'):
        logger.info('Reading data from parquet at %s', path)
        data = pd.read_parquet(path)
    else:
        logger.info('Reading data from csv at %s', path)
        data = pd.read_csv(path)

    logger.info('Data schema:\n%s', data.dtypes)
    logger.info('Data head:\n%s', data.head().to_string())
    data = data.fillna({TARGET: 'UNKNOWN'}) \
        .astype({TARGET: 'category'}) \
        .sort_values(by=[DATETIME_COLUMN, *GROUP_COLUMNS]) \
        .drop(GROUP_COLUMNS, axis=1)

    return data


def get_column_order(data: pd.DataFrame) -> list:
    return data \
        .drop([TARGET, DATETIME_COLUMN], axis=1) \
        .columns \
        .tolist()


def train_model(data: pd.DataFrame, columns: list) -> RandomForestClassifier:
    model = RandomForestClassifier(verbose=1,
                                   n_estimators=9,
                                   max_depth=10,
                                   class_weight='balanced')

    model.fit(data[columns], data[TARGET].cat.codes)
    return model


def get_request_schema(data: pd.DataFrame) -> dict:
    row = data.iloc[:1].astype({DATETIME_COLUMN: str})
    return pynavio.make_example_request(
        row.to_dict(orient='records')[0], TARGET, DATETIME_COLUMN)


def mock_data():
    return pd.DataFrame({
        TARGET: list('abc'),
        'X': [4, 5, 6],
        DATETIME_COLUMN: [1, 2, 3],
        **{key: [1, 1, 1] for key in GROUP_COLUMNS}
    }).astype({TARGET: 'category'})


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, Path]]] = None):

    data = prepare_data('data/activity_recognition.csv')
    column_order = get_column_order(data)
    request_schema = get_request_schema(data)

    model = train_model(data, column_order)

    mlflow_model = Timeseries(data[TARGET].cat.categories.tolist(),
                              column_order, explanations)

    with TemporaryDirectory() as tmp_dir:
        dataset = None
        if with_data:
            data_path = f'{tmp_dir}/timeseries.csv'

            # take only the first 1000 rows to limit archive size
            data.iloc[:1000].to_csv(data_path, index=False)
            dataset = dict(name='timeseries-data', path=data_path)

        model_path = f'{tmp_dir}/model.joblib'
        joblib.dump(model, model_path)

        pip_packages = ['mlflow==2.9.2', 'scikit-learn==1.2.2', 'joblib==1.3.2', 'pynavio==0.2.4']

        if explanations == 'plotly':
            pip_packages.extend(['plotly==5.9.0', 'shap==0.44.1'])

        pynavio.mlflow.to_navio(mlflow_model,
                                example_request=request_schema,
                                artifacts={'model': model_path},
                                dataset=dataset,
                                path=path,
                                code_path=code_path,
                                explanations=explanations,
                                pip_packages=pip_packages,
                                oodd='default' if with_oodd else 'disabled')
