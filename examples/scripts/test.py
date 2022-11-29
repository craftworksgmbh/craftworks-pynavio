import sys
import subprocess
import time
import json
import requests
from pynavio._mlflow import _fetch_data


def _check_model_serving(model_path, port=5001, request_bodies=None):
    URL = f'http://127.0.0.1:{port}/invocations'
    process = subprocess.Popen(
        f'mlflow models serve -m {model_path} -p {port} --no-conda'.split())
    time.sleep(5)
    response = None

    try:
        for data in (request_bodies or _fetch_data(model_path)):
            response = requests.post(
                URL,
                data=json.dumps(data, allow_nan=True),
                headers={'Content-type': 'application/json'})
            response.raise_for_status()
    finally:
        process.terminate()
        if response is not None:
            print(response.json())
        subprocess.run('pkill -f gunicorn'.split())
        time.sleep(2)


if __name__ == '__main__':
    model_path = sys.argv[1]
    _check_model_serving(model_path, port=5001)
