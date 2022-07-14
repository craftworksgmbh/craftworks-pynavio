def test_error_model(helper, tmp_path):
    model_name = 'error_model'
    model_path = str(tmp_path / 'model')
    helper.run(model_name, model_path, expect_error=True)
