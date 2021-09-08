import tempfile
import mlflow.pyfunc

import models.tabular
from test_examples.test_mlflow_models.test_models.tabular_example_request import example_request

PREDICTION = 'prediction'


def test_setup_predict():
    """
    Tests that setup stores a model that can be loaded by mlflow
    """

    with tempfile.TemporaryDirectory() as model_path:
        models.tabular.setup(with_data=False, with_oodd=False, explanations=None, path=model_path)

        model = mlflow.pyfunc.load_model(model_path)

    model_input = example_request
    model_output = model.predict(model_input)

    expecpted_keys = [PREDICTION]

    for key in expecpted_keys:
        assert key in model_output

    for key in expecpted_keys:
        assert len(model_output[key]) == model_input.shape[0]
