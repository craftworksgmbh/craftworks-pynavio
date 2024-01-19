""" This handles tests for all models which are not added to the
    EXCLUDED_MODELS list in conftest.py.

    If the logic in this test is not sufficient for your example model,
    please add it to EXCLUDED_MODELS and define the custom logic in
    the corresponding test_<model-name>.py file.
"""


def test_models_default(model_name, helper, tmp_path):
    model_name = 'minimal_trainer'
    model_path = str(f'/Users/clarareolid/Documents/cw_projects/Navio/repos/craftworks-pynavio-private/examples/NAVIO-25892/{model_name}_final')
    helper(model_name=model_name, model_path=model_path)
