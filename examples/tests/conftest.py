import pytest
from mlflow_models import *
from mlflow_models import __all__ as MODELS
from pynavio._mlflow import ModelValidator, check_model_serving


# these require custom tests - see corresponding test_<model_name>.py
EXCLUDED_MODELS = [
    'car_price_model', 'timeseries', 'visual_inspection_model',
    'timeseries_trainer', 'pump_leakage_model', 'error_model'
]

MODELS = [*filter(lambda model: model not in EXCLUDED_MODELS, MODELS)]


@pytest.fixture(params=MODELS)
def model_name(request):
    return request.param


class Helper(ModelValidator):

    @staticmethod
    def setup_model(model_name, model_path):
        assert model_name in [*MODELS, *EXCLUDED_MODELS]
        import mlflow_models
        setup_arguments = dict(with_data=False,
                               with_oodd=False,
                               explanations=None,
                               path=model_path,
                               code_path=[mlflow_models.__path__[0]])

        globals()[model_name].setup(**setup_arguments)

    @staticmethod
    def verify_model_output(model_output,
                            expect_error=False,
                            model_input=None,
                            **kwargs):
        super(Helper, Helper).verify_model_output(model_output)
        if not expect_error:
            key = 'prediction'
            assert key in model_output

            if model_input is not None:
                # this is not always the case, e.g. some timeseries models
                # will output only one prediction for a set of timeseries rows (frame)
                assert len(model_output[key]) == model_input.shape[0], \
                    'The number of elements in the prediction array must match ' \
                    'the number of input rows'

    @staticmethod
    def verify_model_serving(model_path, port=5001, request_bodies=None):
        try:
            check_model_serving(model_path, port, request_bodies)
        except Exception:
            pytest.fail("Error in the model serving/prediction")

    def __call__(self, model_path, expect_error: bool = False,
                 validate_model_serving=True, validation_port=5001, **kwargs):
        self.setup_model(kwargs["model_name"], model_path)
        model_input, model_output = self.run_model_io(model_path)
        self.verify_model_output(model_output, expect_error=expect_error, model_input=model_input)
        self.verify_model_serving(model_path, validation_port)


@pytest.fixture
def helper():
    return Helper()
