=======
History
=======

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
* attempts to remove the build badgegi
* removes installation for development section form readme
* fixes repository on the pypi page
* adds history
