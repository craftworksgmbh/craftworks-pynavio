# uses code parts from Open Source project
# https://www.kaggle.com/maciejautuch/car-price-prediction
import pathlib
from pathlib import PosixPath
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pynavio
from pynavio import (get_module_path, infer_external_dependencies,
                     make_example_request, prediction_call)
from pynavio.mlflow import to_navio

TARGET = 'target'
PRICE = 'price'
CAT_COLS = [
    'manufacturer', 'cylinders', 'fuel', 'title_status', 'transmission',
    'drive', 'type', 'paint_color', 'condition', 'posting_date', 'state',
    'model', 'region'
]

NUM_COLS = ['year', 'odometer']


class CarPriceModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context) -> None:
        self._model = joblib.load(context.artifacts['model'])
        self._one_hot_enc = joblib.load(context.artifacts['one_hot_enc'])
        self._scaler = joblib.load(context.artifacts['scaler'])
        self._na_fill_values = joblib.load(context.artifacts['na_fill_values'])

    @prediction_call
    def predict(self, context, model_input) -> dict:
        model_input = model_input.fillna(self._na_fill_values)
        model_input = transform(model_input, self._one_hot_enc, self._scaler)
        return pd.Series(self._model.predict(model_input)) \
            .round(2) \
            .pipe(lambda s: {'prediction': s.tolist()})


def transform(X, ohe, scaler):
    X_sc = scaler.transform(X[NUM_COLS])
    X_ohe = ohe.transform(X[CAT_COLS]).toarray()

    return pd.concat(
        [
            pd.DataFrame(
                X_ohe,
                # get new feature names after one hot encoding
                columns=ohe.get_feature_names()),
            pd.DataFrame(X_sc, columns=NUM_COLS)
        ],
        axis=1)


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

    X = pd.concat([numerical, categorical], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=24)
    X_train = pd.DataFrame(X_train, columns=[*NUM_COLS, *CAT_COLS])
    X_test = pd.DataFrame(X_test, columns=[*NUM_COLS, *CAT_COLS])
    ohe = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()
    scaler.fit(X_train[NUM_COLS])
    ohe.fit(X_train[CAT_COLS])

    X_train = transform(X_train, ohe, scaler)
    X_test = transform(X_test, ohe, scaler)

    etr = ExtraTreesRegressor(random_state=0,
                              n_estimators=250,
                              max_features=None,
                              min_samples_split=6)

    etr.fit(X_train, y_train)
    print("train", etr.score(X_train, y_train))
    print("test", etr.score(X_test, y_test))
    return ohe, scaler, na_fill_values, etr


def _load_data():
    current_path = pathlib.Path(__file__).parent.resolve()
    # downloaded the data from
    # https://www.kaggle.com/austinreese/craigslist-carstrucks-data
    data_path = current_path / 'data' / 'vehicles.csv'
    df = pd.read_csv(data_path, nrows=10000)
    return df


def load_data():
    df = _load_data()
    y = df[PRICE]
    X = df.drop([PRICE], axis=1)

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
        data[[*NUM_COLS, *CAT_COLS, TARGET]].to_dict(orient='records')[91],
        TARGET)

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
                *infer_external_dependencies(get_module_path(
                    pynavio))  # TODO: rm this in the final example of using
                # installed pynavio lib, as this is a dependency of pynavio
            ]))

        to_navio(CarPriceModel(),
                 example_request=example_request,
                 explanations=explanations,
                 artifacts={
                     'model': model_path,
                     'scaler': scaler_path,
                     'one_hot_enc': one_hot_enc_path,
                     'na_fill_values': na_fill_values_path
                 },
                 path=path,
                 pip_packages=pip_packages,
                 code_path=code_path,
                 dataset=dataset,
                 oodd='default' if with_oodd else 'disabled')
