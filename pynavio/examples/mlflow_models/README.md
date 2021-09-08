# Example MLFlow Models

Available models and their respective data sets:

- tabular - iris.csv (available via scikit-learn)

## Setup

1. Create a directory named `data` in this folder
2. Copy `activity_recognition.csv` into `data`
3. Create a python (v3.7 or above) env via:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Creating Models

Make sure the python env is active (i.e. `source venv/bin/activate` has been run)
and run:

```sh
make
```

The created zip files in `model_files` directory are the usable MLFlow models.

## Plotly Explanations

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
