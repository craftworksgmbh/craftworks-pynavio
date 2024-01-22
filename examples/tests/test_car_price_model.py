from mlflow_models import car_price_model


def test_car_price_model(helper, tmp_path, monkeypatch):
    model_name = 'car_price_model'
    model_path = str(tmp_path / 'model')
    monkeypatch.setattr(car_price_model, "_load_data",
                        car_price_model.mock_data)
    helper(model_name=model_name, model_path=model_path)
