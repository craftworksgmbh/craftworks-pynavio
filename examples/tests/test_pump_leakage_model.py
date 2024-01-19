from mlflow_models import pump_leakage_model


def test_pump_leakage_model(helper, tmp_path, monkeypatch):
    model_name = 'pump_leakage_model'
    model_path = str(f'/Users/clarareolid/Documents/cw_projects/Navio/repos/craftworks-pynavio-private/examples/NAVIO-25892/{model_name}_re')
    monkeypatch.setattr(pump_leakage_model, "load_data",
                        pump_leakage_model.mock_data)
    helper(model_name=model_name, model_path=model_path)
