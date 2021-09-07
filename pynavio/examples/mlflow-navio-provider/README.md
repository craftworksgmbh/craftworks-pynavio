
## Files

* mlflow_environment.yaml :: conda environment for executing Navio-MLFlow-Model-Serving-Tutorial.ipynb and building the container
* mlflow-template :: its content is the output of executing Navio-MLFlow-Model-Serving-Tutorial.ipynb, i.e. the MLflow Model
* Navio-MLFlow-Model-Serving-Tutorial.ipynb :: data scientist does the modeling and uses Mlflow Models to package
* README.txt :: readme
* temp/example_request.json :: tmp output of Navio-MLFlow-Model-Serving-Tutorial.ipynb
* temp/scaler.pkl :: tmp output of Navio-MLFlow-Model-Serving-Tutorial.ipynb
* temp/xgb_model.pkl :: tmp output of Navio-MLFlow-Model-Serving-Tutorial.ipynb



## How to containerize using MLflow?

Setup environment:

    conda env create -f mlflow_environment.yaml
    conda activate mlflow


From this directory...:

    mlflow models build-docker --model-uri mlflow-template/mlflow-custom-pipeline --install-mlflow
    docker run -p 5001:8080 "mlflow-pyfunc-servable:latest"
    curl -H "Content-Type: application/json" -X POST http://localhost:5001/invocations -d '{
    "columns": ["x1", "x2", "x3", "x4"],
    "data": [[4.1, 5.1, 6.1, -4.1], [4.1, 5.1, 6.1, -4.1]]
    }'


--> Navio only requires the zipped file mlflow-template/mlflow-custom-pipeline.zip

## Troubleshooting
* bash: gunicorn: command not found
    * --install-mlflow missing or mlflow not installed in conda.yaml