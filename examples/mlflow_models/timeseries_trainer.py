import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn

import pynavio

from .timeseries import (TARGET, Timeseries, get_column_order,
                         get_request_schema, prepare_data, train_model)


class TimeseriesTrainer(mlflow.pyfunc.PythonModel):

    @staticmethod
    def _read_config(model_input: pd.DataFrame) -> tuple:
        logger = logging.getLogger('gunicorn.error')
        row = model_input.iloc[0]
        data_path, dest_path = map(Path, [row.dataPath, row.destinationPath])

        if not data_path.is_dir():
            logger.info(f'Data path is a file: {data_path.is_file()}')
        else:
            logger.info('Data path contents:\n' +
                        '\n'.join(map(str, data_path.glob('*'))))

        if dest_path.suffix != '.zip':
            logger.info('Destination path does not have a .zip suffix. '
                        'Treating as directory')
        else:
            dest_path = dest_path.parent / dest_path.stem
            logger.info(f'Model will be written to {dest_path} and saved as '
                        f'{dest_path}.zip')

        return data_path, dest_path

    @pynavio.prediction_call
    def predict(self, context, model_input):
        logger = logging.getLogger('gunicorn.error')
        logger.info(f'Received config:\n{model_input.iloc[0]}')
        data_path, destination_path = self._read_config(model_input)

        logger.info('Training...')

        data = prepare_data(str(data_path))
        column_order = get_column_order(data)
        request_schema = get_request_schema(data)

        model = train_model(data, column_order)
        mlflow_model = Timeseries(data[TARGET].cat.categories.tolist(),
                                  column_order)

        import mlflow_models

        pip_packages = ['mlflow==2.9.2', 'scikit-learn==1.4.0', 'joblib==1.3.2', 'pynavio==0.3.1']

        with TemporaryDirectory() as tmp_dir:
            model_path = f'{tmp_dir}/model.joblib'
            joblib.dump(model, model_path)

            zip_path = pynavio.mlflow.to_navio(
                mlflow_model,
                example_request=request_schema,
                artifacts={'model': model_path},
                pip_packages=pip_packages,
                code_path=[mlflow_models.__path__[0], pynavio.__path__[0]],
                path=destination_path)

        logger.info(f'Done training. Model stored as {zip_path} in '
                    f'{zip_path.parent.absolute()}')
        return {'prediction': [str(zip_path.absolute())]}


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, Path]]] = None):

    with TemporaryDirectory() as tmp_dir:
        pip_packages = ['mlflow==2.9.2', 'scikit-learn==1.4.0', 'joblib==1.3.2', 'pynavio==0.3.1']
        example = pynavio.make_example_request(
            {
                'dataPath': 'mlflow-models/data/activity_recognition.parquet',
                'destinationPath': './model.zip',
                'zipAbsPath': 'abc/xyz',
            },
            target='zipAbsPath')
        pynavio.mlflow.to_navio(TimeseriesTrainer(),
                                example_request=example,
                                path=path,
                                pip_packages=[*pip_packages, 'pyarrow'],
                                code_path=code_path,
                                oodd='default' if with_oodd else 'disabled')
