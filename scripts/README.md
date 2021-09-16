# pynavio Scripts

## Example Models

Available models and their respective data sets:

- tabular - iris.csv (available via scikit-learn)

To build the models, run:

```sh
make
```

The created zip files in `model_files` directory are the usable MLFlow models.

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
