import sys

from pynavio._mlflow import _check_model_serving


if __name__ == '__main__':
    model_path = sys.argv[1]
    _check_model_serving(model_path, port=5001)
