import tempfile
import mlflow.pyfunc
import pandas as pd
import pytest

from models import tabular
from scripts.test import _fetch_data, _check_model_serving

PREDICTION = 'prediction'


def _get_example_request_df(model_path):
    data = _fetch_data(model_path)[0]
    return pd.DataFrame(data['data'], columns=data['columns'])


def test_setup_predict():
    """
    Tests that setup stores a model that can be loaded by mlflow
    """

    with tempfile.TemporaryDirectory() as model_path:
        tabular.setup(with_data=False, with_oodd=False, explanations=None, path=model_path)

        model = mlflow.pyfunc.load_model(model_path)
        model_input = _get_example_request_df(model_path)

        # sanity-check the loaded model
        model_output = model.predict(model_input)
        expected_keys = [PREDICTION]

        for key in expected_keys:
            assert key in model_output
            assert len(model_output[key]) == model_input.shape[0]

        # check model serving/prediction
        try:
            _check_model_serving(model_path)
        except Exception:
            pytest.fail("Error in the model serving/prediction")

