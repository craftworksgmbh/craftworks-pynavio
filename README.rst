=======
pynavio
=======


.. image:: https://img.shields.io/pypi/v/pynavio.svg
        :target: https://pypi.python.org/pypi/pynavio

.. image:: https://img.shields.io/travis/craftworksgmbh/pynavio.svg
        :target: https://travis-ci.com/craftworksgmbh/pynavio

.. image:: https://readthedocs.org/projects/pynavio/badge/?version=latest
        :target: https://pynavio.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Python lib for navio


* Free software: MIT license


Features
--------

* Pynavio.mlflow.to_navio function calls mlflow.pyfunc.save_model function, saving a model zip file as required by navio.
* Pynavio.infer_external_dependencies is a helper function that infers the external dependencies based on the file path. For its limitations please refer to its doc string.
* Pynavio.infer_imported_code_path is a helper function that  infers the imported code paths based on the file path and the root path. For its limitations please refer to its doc string.
* Pynavio.make_example_request generates a request schema for a navio model from data.



Installation for development
============================

* Create dedicated virtual environment: conda create --name pynavio pip python=3.6
* run::

    $ TODO: add git repo here
    $ make install (Uses pip to install package only in environment)

* To install requirements necessary for development run::

    $ pip install -r requirements_dev.txt


Versioning
==========

Run (replace "part" with either major, minor or patch)::

    $ bumpversion part

Deploying
==========

Run (replace part with either major, minor or patch)::

    $ bumpversion part
    $ git push
    $ git push --tags
    $ make release

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


