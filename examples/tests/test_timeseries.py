from mlflow_models import timeseries


def test_timeseries(helper, tmp_path, monkeypatch):
    model_name = 'timeseries'
    model_path = str(f'/Users/clarareolid/Documents/cw_projects/Navio/repos/craftworks-pynavio-private/examples/NAVIO-25892/{model_name}_re')
    monkeypatch.setattr(timeseries, "prepare_data",
                        lambda _: timeseries.mock_data())
    helper(model_name=model_name, model_path=model_path)
