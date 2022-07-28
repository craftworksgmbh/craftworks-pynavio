# pynavio Example Models

Available models and their respective data sets:

- tabular - iris.csv (available via scikit-learn)
- car_price_model - needs to be manually downloaded the data from https://www.kaggle.com/austinreese/craigslist-carstrucks-data into examples/data/vehicles.csv
- pump_leakage_models - is automatically downloaded from 'https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip'
- visual_inspection_model - donwload manually [https://doi.org/10.5281/zenodo.4694694](https://doi.org/10.5281/zenodo.4694694)

Note: car_price_model and pump_leakage_models do not support plotly explanations.

## Setup

- Use `python >= 3.8`
- Run `pip install -r requirements.txt && pip install -r requirements_dev.txt` from the repo root directory

## Running

To build all models, run:

```sh
make
```

To build a specific model:

```sh
make <model-name>

# example
make image
```

The created zip files in `model_files` directory are the usable MLflow models. Once built, the model archive
is also tested by querying its predictor with the provided example data.

For a listing of available models, run:

```sh
make help
```

### Naming Convention

All model names have the structure `<name><suffix>`, where `<suffix>` follows the regex

```
(_data)?(_oodd)?(_plotly|_default)?$
```

- `data` indicates that the model archive will contain a dataset
- `oodd` indicates that the model has OOD enabled
- `default` indicates that the model supports default navio explanations
- `plotly` indicates the the model generates custom explanations via plotly

## Unit Tests

There is one test case for each model. Each test case performs:

1. Model setup
2. Model loading
3. Call model predict method on example data
4. Check model serving via `mlflow models serve`

Many of the example model generation scripts are self-contained - they do not require any external data sets or input.
Test cases for these models are generated automatically - see `test_models_default.py` and the fixtures in `conftest.py`.

In other cases, it is necessary to mock data loading or other parts of the model setup logic. In such cases:

- Add the name of the model to `EXCLUDED_MODELS` list in `conftest.py`
- Add a `test_<model-name>.py` with the custom logic, as done in e.g. `test_timeseries.py`

### Models with Plotly Explanations

Explanations are generated when the following conditions are fulfilled

- The model has a metadata field "explanations.format" with value "plotly"
- There is a boolean column `is_background` in the model input (request)
- There is at least one row with value `true` in the `is_background` column
- There is at least one row with value `false` in the `is_background` column

Normally (without explanations), the model will produce the following response:

```
{
    "prediction": [<value-as-string>, ...],
}
```

When conditions for explanations are fulfilled, the response will look like:

```
{
    "prediction": [<value-as-string>, ...],
    "explanation": [<plotly-figure-json>, ...]
}
```

Note that some models (not given in these examples) may only output 1 prediction
for several rows, in which case the prediction and explanation arrays will have
1 element even if multiple data rows are provided in the request.
