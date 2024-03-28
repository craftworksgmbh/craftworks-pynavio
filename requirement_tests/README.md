# Tests agains different python versions
The current approach for testing the different dependencies is based on tox, which is an environment orchestrator that allows checking that the different packages/dependencies build and are installed correctly under different environments. It allows to create a file that contains an `envlist`  where you specify the dependencies and version ranges as well as the `python` environment version you want to test.

An example of how it can be done:

```
envlist = {py37}-Pillow{800,801,810,811,812,820,830,831,832,840,900,901,910,911,920,930,940,950,1000,1001,1010,1020},
          {py37}-Plotly{4100,4110,4120,4130,4140,4141,4142,4143,500,510,521,522,530,531,540,550,560,570,580,581,582,590,5100,5110,5120,5130,5131,5140,5141,5150,5160,5161,5170,5180},
          {py37}-Pigar{090,091,092,100,101,102,200,201,202,203,204,205,206,207,208,212,211},flake8
```

The way that tox works is the following, when specifying the `python` version tox looks in the env that you are running the file on a python interpreter that matches the version you have specified. Hence, you will need a env for each of the python versions you want to test. This means that for testing with the current approach, you will need to change the `{py37}` in the `enlist` for the python version `{pyXY}` that you need to run the tests in. The output tox will give is a list with the dependencies and versions that you have stablished and the state of the test, being `succeed` if it ha passed and some error if not.
