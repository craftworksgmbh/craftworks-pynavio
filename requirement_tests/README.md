# Tests agains different python versions
The current approach for testing the different dependencies is based on a tox automation process, which is a tool that allows checking that the different packages/dependencies build and are installs correctly under different environments. It is described as a environment orchestrator, which is done by creating a file that contains an `enlist`  where you specifiy the dependencies and ranges you want to test and the `python` environment version against which you want to apply the tests.

An example of how it can be done:

```
envlist = {py37}-Pillow{800,801,810,811,812,820,830,831,832,840,900,901,910,911,920,930,940,950,1000,1001,1010,1020},
          {py37}-Plotly{4100,4110,4120,4130,4140,4141,4142,4143,500,510,521,522,530,531,540,550,560,570,580,581,582,590,5100,5110,5120,5130,5131,5140,5141,5150,5160,5161,5170,5180},
          {py37}-Pigar{090,091,092,100,101,102,200,201,202,203,204,205,206,207,208,212,211},flake8
```

The way that tox works is the following, when specifying the `python` version tox looks in the env that you are running the file on a python interpreter that matches the version you have specified. Hence, you will need a env for each of the python versions you want to test. This means that for testing with the current approach, you will need to change the `{py37}` in the `enlist` for the python version `{pyXY}` that you need to run the tests in. The output tox will give is a list with the dependencies and versions that you have stablished and the state of the test, being `succeed` if it ha passed and some error if not.

## Files
In this folder we have the testing files:
* tox.ini: It tests the mlflow versions against the python versions as it is widely used in pynavio
* tox_version_packages.ini: This file contains the test of three libraries used by pynavio (Plotly, Pigar and Pillow)

and the requirement files:
* requirements_tox.txt: Contains the necessary requirements to run the test for the tox setup
