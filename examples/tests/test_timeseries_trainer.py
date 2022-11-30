import pandas as pd
from mlflow_models import timeseries


def _prepare_test(path):
    data_path = path / 'data.csv'
    data = timeseries.mock_data()
    data.to_csv(data_path, index=False)
    body = {
        'destinationPath': str(path / 'tmp_model.zip'),
        'dataPath': str(data_path.absolute())
    }
    return {'columns': list(body.keys()), 'data': [list(body.values())]}


def test_timeseries_trainer(helper, tmp_path, monkeypatch):
    model_name = 'timeseries_trainer'
    model_path = str(tmp_path / 'model')

    request = _prepare_test(tmp_path)
    model_input = pd.DataFrame(**request)

    helper.setup_model(model_name, model_path)
    model_input, model_output = helper.run_model_io(model_path, model_input)
    helper.verify_model_output(model_output)

    helper.verify_model_serving(model_path, request_bodies=[request])
