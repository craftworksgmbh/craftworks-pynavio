=======
pynavio
=======


.. image:: https://img.shields.io/pypi/v/pynavio.svg
        :target: https://pypi.python.org/pypi/pynavio

Python lib for navio

* Free software: MIT license


Features
--------
* Pynavio.Client, navio API client, allows to upload models and data, deploy and retrain models etc.
* Pynavio.mlflow.to_navio function calls mlflow.pyfunc.save_model function, saving a model zip file as required by navio.
* Pynavio.infer_external_dependencies is a helper function that infers the external dependencies based on the file path. For its limitations please refer to its doc string.
* Pynavio.infer_imported_code_path is a helper function that  infers the imported code paths based on the file path and the root path. For its limitations please refer to its doc string.
* Pynavio.make_example_request generates a request schema for a navio model from data.

Documentation
-------------

The official documentation is hosted on https://navio.craftworks.io : https://navio.craftworks.io/docs/guides/pynavio/


Examples
==========

To build all example models, use::

    $ cd scripts && make


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Examples/models uses code parts from Open Source project `mjain72/Condition-monitoring-of-hydraulic-systems-using-xgboost-modeling`_.

.. _`mjain72/Condition-monitoring-of-hydraulic-systems-using-xgboost-modeling`: https://github.com/mjain72/Condition-monitoring-of-hydraulic-systems-using-xgboost-modeling

Examples/models uses code parts from Open Source project `https://www.kaggle.com/maciejautuch/car-price-prediction`_

.. _`https://www.kaggle.com/maciejautuch/car-price-prediction`: https://www.kaggle.com/maciejautuch/car-price-prediction


