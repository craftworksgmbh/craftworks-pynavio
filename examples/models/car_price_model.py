import json

import joblib
import mlflow
import os
import kaggle
import subprocess
import pandas as pd
from typing import List, Optional, Union
from pathlib import PosixPath, Path
from tempfile import TemporaryDirectory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import pynavio
from pynavio.utils.common import (get_module_path, make_example_request,
                                  prediction_call, to_navio_mlflow)
from pynavio.utils.infer_dependencies import infer_external_dependencies
from sklearn.ensemble import ExtraTreesRegressor
TARGET = 'target'

CAT_COLS = ['manufacturer', 'cylinders', 'fuel', 'title_status', 'transmission',
            'drive', 'type', 'paint_color', 'condition', 'posting_date',
            'state', 'model', 'region']

NUM_COLS = ['year', 'odometer']


class CarPriceModel(mlflow.pyfunc.PythonModel):
    def __init__(self, columns):
        self._columns = columns

    def load_context(self, context) -> None:
        self._model = joblib.load(context.artifacts['model'])
        self._one_hot_enc = joblib.load(context.artifacts['one_hot_enc'])
        self._scaler = joblib.load(context.artifacts['scaler'])
        self._na_fill_values = joblib.load(context.artifacts['na_fill_values'])

    def _predict(self, model_input) -> list:
        # fill values
        model_input = model_input.fillna(self._na_fill_values)
        return pd.Series(
                self._model.predict(transform(model_input, self._one_hot_enc, self._scaler))) \
            .pipe(lambda s: {'prediction': s.tolist()})

    @prediction_call
    def predict(self, context, model_input):
        return self._predict(model_input)


def transform(X, ohe, scaler):
    X_sc = scaler.transform(X[NUM_COLS])
    X_ohe = ohe.transform(X[CAT_COLS]).toarray()

    return pd.concat(
        [pd.DataFrame(X_ohe, columns=ohe.get_feature_names()),
         pd.DataFrame(X_sc, columns=NUM_COLS)], axis=1
    )


def train_car_price_model(X, y):
    # separate categorical cols needed for training
    # other cols have many missing values and/or outliers
    categorical = X[CAT_COLS]
    numerical = X[NUM_COLS]

    na_fill_values = dict()
    # fill categorical with mode
    for cat in categorical:
        na_fill_values[cat] = categorical[cat].mode().values[0]
        categorical[cat] = categorical[cat].fillna(na_fill_values[cat])

    for num in numerical:
        na_fill_values[num] = numerical[num].mean()
        numerical[num] = numerical[num].fillna(na_fill_values[num])

    X = pd.concat(
        [numerical, categorical], axis=1
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    X_train = pd.DataFrame(X_train, columns=[*NUM_COLS, *CAT_COLS])
    X_test = pd.DataFrame(X_test, columns=[*NUM_COLS, *CAT_COLS])
    ohe = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()
    scaler.fit(X_train[NUM_COLS])
    ohe.fit(X_train[CAT_COLS])

    X_train = transform(X_train, ohe, scaler)
    X_test = transform(X_test, ohe, scaler)

    etr = ExtraTreesRegressor(random_state=0, n_estimators=250, max_features=None, min_samples_split=6)

    etr.fit(X_train, y_train)
    print("train", etr.score(X_train, y_train))
    print("test", etr.score(X_test, y_test))
    return ohe, scaler, na_fill_values, etr


def load_data():

    with TemporaryDirectory() as tmp_dir:
        data_path = f'{tmp_dir}/data_path'
        Path(data_path).mkdir()
        my_env = os.environ.copy()
        my_env['KAGGLE_USERNAME'] = 'cwghar'
        my_env['KAGGLE_KEY'] = 'd93f82fb339265fcf7bd76c013565ba2'
        result = subprocess.check_call(
            (
                # credentials for kaggle api
                'env', 'KAGGLE_USERNAME=cwghar', 'KAGGLE_KEY=d93f82fb339265fcf7bd76c013565ba2',
                'kaggle', 'datasets', 'download', 'austinreese/craigslist-carstrucks-data',
                '--unzip', '--path', f'{data_path}',
              ),
            env=my_env

        )
        if result != 0:
            print("error during downloading the dataset")
            raise AssertionError

        df = pd.read_csv(f'{data_path}/vehicles.csv',
                         nrows=100)  # TODO: increase it to 10 000, set to 100 for speed of testing
    y = df['price']
    X = df.drop(['price'], axis=1)
    return X, y


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, PosixPath]]] = None):

    X, y = load_data()

    # make example request
    data = X.copy()
    data[TARGET] = y

    example_request = make_example_request(
        data[[*NUM_COLS, *CAT_COLS, TARGET]].to_dict(orient='records')[91], TARGET)

    one_hot_enc, scaler, na_fill_values, model = train_car_price_model(X, y)

    with TemporaryDirectory() as tmp_dir:
        model_path = f'{tmp_dir}/model.joblib'
        joblib.dump(model, model_path)

        one_hot_enc_path = f'{tmp_dir}/one_hot_enc.joblib'
        joblib.dump(one_hot_enc, one_hot_enc_path)

        scaler_path = f'{tmp_dir}/scaler.joblib'
        joblib.dump(scaler, scaler_path)

        na_fill_values_path = f'{tmp_dir}/na_fill_values.joblib'
        joblib.dump(na_fill_values, na_fill_values_path)

        dataset = None
        if with_data:
            data_path = f'{tmp_dir}/car_price.csv'
            data.to_csv(data_path, index=False)
            dataset = dict(name='car_price-data', path=data_path)

        pip_packages = list(
            set([
                *infer_external_dependencies(__file__),
                *infer_external_dependencies(
                    get_module_path(pynavio)
                )  #TODO: rm this in the final example of using installed pynavio lib, as this is a dependency of pynavio
            ]))

        to_navio_mlflow(CarPriceModel([*NUM_COLS, *CAT_COLS]),
                        example_request=example_request,
                        explanations=explanations,
                        artifacts={'model': model_path,
                                   'scaler': scaler_path,
                                   'one_hot_enc': one_hot_enc_path,
                                   'na_fill_values': na_fill_values_path},
                        path=path,
                        pip_packages=pip_packages,
                        code_path=code_path,
                        dataset=dataset,
                        oodd='default' if with_oodd else 'disabled')
