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
* Documentation: https://pynavio.readthedocs.io.


Features
--------

* TODO

Installation for development
============================

* Create dedicated virtual environment: conda create --name pynavio pip python=3.6
* run::

    $ git clone git@github.com:craftworksgmbh/craftworks-pynavio.git
    $ make install (Uses pip to install package only in environment)

* To install requirements necessary for development run::

    $ pip install -r requirements_dev.txt

* Setup Nexus PyPi (pipy@nexus)for uploading releases:
    * Open pypirc file::

        $ nano ~/.pypirc


    * and  insert::

        [distutils]
        index-servers =
        nexus
        [nexus]
        repository: https://nexus.craftworks.io/repository/pypi-internal/
        username: pipy-user
        password: <available in 1PW>

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
