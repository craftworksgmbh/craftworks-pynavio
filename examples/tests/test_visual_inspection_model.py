from mlflow_models.visual_inspection_model import mock_data, train


def test_visual_inspection_model(helper, tmp_path, monkeypatch):
    model_name = 'visual_inspection_model'
    model_path = str(tmp_path / 'model')

    _train = train.train

    def _mock_train(path: str):
        return _train(path, epochs=1)

    monkeypatch.setattr(train, "setup_data", mock_data)
    monkeypatch.setattr(train, "train", _mock_train)

    helper.run(model_name, model_path)
