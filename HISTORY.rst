=========
History
=========

0.0.1 (2021-09-07)
------------------

* First release on PyPI.

0.0.2 (2021-12-29)
------------------

* allows user provide modules as pip strings
* brings possibility to infer package dependencies(pip strings) using the infer_external_dependencies()
* handle data frame input for example request
* updates the default conda channels: remove anaconda and pytorch and add conda-forge
* adds and updates the dependencies
* brings gpu related changes
* brings example models (tabular etc.)
* fixes the issue with pigar running for 10 minutes when the input path contains venv

0.1.0 (2022-01-24)
------------------

* adds model helpers
* brings more example models: pump leakage model
* removes the need to download the data from kaggle, refers to local data dir for data instead
* adds credits to README, adds LICENSE

0.1.1 (2022-02-07)
------------------

* adds credits for kaggle notebook's code parts usage
* brings more example models: car price model

0.1.4 (2022-08-25)
------------------

* API Client
* adds possibility to infer the dependencies used in the file only
* extra docs for example models
* moves the data dir out into examples
* updates dependencies
* fixes the missing import of the currently available models
* fixes mlflow==1.27.0 circular import error
* fixes pynavio FileNotFound error mlflow-1.26.0
* fixes _fetch_data example request path according to mlflow-1.26.0
* sets protobuf version to fix failing tests
* fixes to_navio() to support kedro path

0.1.5 (2022-09-15)
------------------

* fixes the documentation link
* attempts to remove the build badge
* removes installation for development section form readme
* fixes repository on the pypi page
* adds history

0.1.6 (2022-09-15)
------------------

* final touches/refactoring to make the repo ready for becoming public

0.1.7 (2022-11-16)
------------------

* fixes mlflow version to be <2.0 as it introduces breaking changes in the api
* wraps navio's model deletion api


0.2.0 (2022-11-30)
------------------

* resolves mlflow2 incompatibility
* brings model validation checks:prediction, example request and MLmodel metadata schema validation
* makes check_model_serving function public
* adds jsonschema to requirements
* bump tensorflow from 2.9.1 to 2.9.3
* bump pillow from 9.2.0 to 9.3.0
* refactors example tests

0.2.1 (2022-12-07)
------------------

* loosens version requirement for jsonschema
* improves and fixes typos in error message texts
* prevents the exception traceback from prediction_call when checking if it is used, to not confuse the user
* brings a descriptive error if model path in code_path
* brings a descriptive error if code_path is not a list

0.2.2 (2023-04-03)
------------------

* bump tensorflow from 2.9.3 to 2.11.1
* updates torch version and adds onnx as requirement
* fixes torch, pyarrow and onnxrutime versions
* fixes the prediction schema to only expect a list as prediction
* bump ipython from 8.4.0 to 8.10.0
* improves the docstring of to_navio function
* bump wheel from 0.37.1 to 0.38.1

0.2.3 (2023-08-08)
------------------

* adds warnings related to limitations for nested inputs and big model.zip size
* make ModelValidator a public class
* adds docstrings to ModelValidator
* makes model validation optional in to_navio(), so one can disable it
* adds (pynavio model validation) to the validation messages so it is clear where they come from
* adds messages for pynavio model validation checks failed(how to disable)/succeeded(how to check model serving)
* updates readme with model validation info

0.2.4 (2023-08-22)
------------------

* add possibility for user to add sys dependencies to the navio model

0.3.0 (2024-01-17)
------------------

* allows for a conda environment with pip requirements to be inferred
* adds possibility to add extra pip dependencies to the inferred environment

0.3.1 (2024-01-23)
------------------

* minor reformatting fix

0.3.2 (2024-05-21)
------------------

* bump Pillow from 9.3.0 to >=8.0.0, <11.0.0
* bump plotly from 5.9.0 to >=4.10.0, <6.0.0

0.3.3 (2025-07-11)
------------------

* adds optional metadata argument to to_navio function to be able to add custom metadata to the MLmodel file
* updates tox configuration to test compatibility across multiple mlflow versions
* updates requirements.txt to restrict mlflow<3
* updates requirements_dev.txt to avoid errors related to numpy>=2 and packaging>=25
