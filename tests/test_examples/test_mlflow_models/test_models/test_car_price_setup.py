import tempfile

import mlflow.pyfunc
import pandas as pd
import pytest

from examples.models import car_price_model
from examples.models.car_price_model import PRICE, NUM_COLS, CAT_COLS
from pynavio import infer_imported_code_path
from scripts.test import _check_model_serving, _fetch_data

PREDICTION = 'prediction'


def mock_data():
    print("Generating dummy data for the test purposes...")
    mock_df = pd.DataFrame({
        PRICE: [5000] * 100,
        **{num_col: [2000] * 100 for num_col in NUM_COLS},
        **{cat_col: ["good state"] * 100 for cat_col in CAT_COLS}
    })
    return mock_df


def _get_example_request_df(model_path):
    data = _fetch_data(model_path)[0]
    return pd.DataFrame(data['data'], columns=data['columns'])


def test_setup_predict(rootpath, monkeypatch):
    """
    Tests that setup stores a model that can be loaded by mlflow
    """
    monkeypatch.setattr(car_price_model, "_load_data", mock_data)
    code_path = infer_imported_code_path(__file__, rootpath)

    with tempfile.TemporaryDirectory() as model_path:
        car_price_model.setup(with_data=False,
                              with_oodd=False,
                              explanations=None,
                              path=model_path,
                              code_path=code_path)

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
