from mlflow_models import timeseries


def test_timeseries(helper, tmp_path, monkeypatch):
    model_name = 'timeseries'
    model_path = str(tmp_path / 'model')
    monkeypatch.setattr(timeseries, "prepare_data",
                        lambda _: timeseries.mock_data())
    helper.run(model_name, model_path)
