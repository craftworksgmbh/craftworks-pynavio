# uses code parts from Open Source project
# https://github.com/mjain72/Condition-monitoring-of-hydraulic-systems-using-xgboost-modeling
import os
import zipfile
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from requests import get
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pynavio

TARGET = 'target'


class PumpLeakageModlel(mlflow.pyfunc.PythonModel):

    def __init__(self, class_mapping, column_order):
        self._class_mapping = class_mapping
        self._column_order = column_order

    def load_context(self, context) -> None:
        self._model = joblib.load(context.artifacts['model'])
        self._scaler = joblib.load(context.artifacts['scaler'])

    def _predict(self, model_input) -> list:
        return pd.Series(
                self._model.predict(self._scaler.transform(
                    model_input[self._column_order]))) \
            .map(self._class_mapping) \
            .pipe(lambda s: {'prediction': s.tolist()})

    @pynavio.prediction_call
    def predict(self, context, model_input):
        return self._predict(model_input)


def read_file(dir_path, filename):
    return pd.read_csv(os.path.join(dir_path, filename), sep='\t', header=None)


def get_df(dir_path):
    feature_files = [
        'PS1.txt', 'PS2.txt', 'PS3.txt', 'PS4.txt', 'PS5.txt', 'PS6.txt',
        'FS1.txt', 'FS2.txt', 'TS1.txt', 'TS2.txt', 'TS3.txt', 'TS4.txt',
        'EPS1.txt', 'VS1.txt', 'CE.txt', 'CP.txt', 'SE.txt'
    ]
    feat_dfs = []
    for file_name in feature_files:
        feat_df = read_file(dir_path=dir_path, filename=file_name)

        feat_dfs.append(
            pd.DataFrame(
                # take the cycle data mean
                feat_df.mean(axis=1),
                columns=[f"{file_name.split('.')[0]}_mean"]))

    X = pd.concat(feat_dfs, axis=1)

    profile = read_file(dir_path=dir_path, filename='profile.txt')
    y = pd.Series(profile.iloc[:, 2])

    return X, y


def train_pump_performance_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=24)

    scl = StandardScaler()

    scl.fit(X_train)
    X_train_s = scl.transform(X_train)

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_s, y_train)
    return scl, lr


def load_data():
    with TemporaryDirectory() as tmp_dir:
        data_path = f'{tmp_dir}/data_path'
        Path(data_path).mkdir()
        extracted_data = Path(data_path) / 'extracted'
        extracted_data.mkdir()

        # download data
        # https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip'
        file_name = f'{tmp_dir}/data_path/data.zip'
        # open in binary mode
        with open(file_name, "wb") as file:
            response = get(url)
            file.write(response.content)

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(extracted_data)

        X, y = get_df(extracted_data)

    return X, y


def mock_data():
    columns = list('abcde')
    num_cols = len(columns)
    target = np.repeat([0, 1, 2], 10)
    return pd.DataFrame(np.random.rand(target.shape[0], num_cols),
                        columns=columns), pd.Series(target)


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, PosixPath]]] = None):
    X, y = load_data()
    class_mapping = {0: "no_leakage", 1: "weak_leakage", 2: "severe_leakage"}
    column_order = X.columns.tolist()

    # make example request
    data = X.copy()
    data[TARGET] = y.map(class_mapping).copy().to_numpy()

    example_request = pynavio.make_example_request(
        data.to_dict(orient='records')[0], TARGET)

    scaler, model = train_pump_performance_model(X, y)

    with TemporaryDirectory() as tmp_dir:
        model_path = f'{tmp_dir}/model.joblib'
        joblib.dump(model, model_path)

        scaler_path = f'{tmp_dir}/scaler.joblib'
        joblib.dump(scaler, scaler_path)

        dataset = None
        if with_data:
            data_path = f'{tmp_dir}/pump_leakage.csv'
            data.to_csv(data_path, index=False)
            dataset = dict(name='pump_leakage-data', path=data_path)

        pynavio.mlflow.to_navio(PumpLeakageModlel(class_mapping, column_order),
                                example_request=example_request,
                                explanations=explanations,
                                artifacts={
                                    'model': model_path,
                                    'scaler': scaler_path
                                },
                                path=path,
                                pip_packages=['scikit-learn==1.3.2', 'pynavio==0.2.4'],
                                code_path=code_path,
                                dataset=dataset,
                                oodd='default' if with_oodd else 'disabled')
