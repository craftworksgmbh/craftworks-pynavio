import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from utils.common import (get_module_path, make_example_request,
                          prediction_call, to_mlflow)

from .minimal import TARGET, Minimal, example_request


class MinimalTrainer(mlflow.pyfunc.PythonModel):

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

    @prediction_call
    def predict(self, context, model_input):
        logger = logging.getLogger('gunicorn.error')
        logger.info(f'Received config:\n{model_input.iloc[0]}')
        data_path, destination_path = self._read_config(model_input)

        logger.info('Training...')

        import models
        import utils
        zip_path = to_mlflow(
            Minimal(),
            example_request,
            dependencies=[np, pd, *map(get_module_path, [models, utils])],
            path=destination_path)

        logger.info(f'Done training. Model stored as {zip_path} in '
                    f'{zip_path.parent.absolute()}')
        return {'prediction': [str(zip_path.absolute())]}


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None):

    with TemporaryDirectory() as tmp_dir:
        model = MinimalTrainer()
        dependencies = [np, pd, './models', './utils']
        example = make_example_request(
            {
                'dataPath': './data/',
                'destinationPath': './model.zip',
                'zipAbsPath': 'abc/xyz',
            },
            target='zipAbsPath')
        to_mlflow(model,
                  example,
                  path=path,
                  dependencies=dependencies,
                  oodd='default' if with_oodd else 'disabled')
