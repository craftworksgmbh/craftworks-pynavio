import base64
import os
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import mlflow
import numpy as np
import onnxruntime
import pandas as pd
import PIL
from PIL import Image

import pynavio


class ImageModel(mlflow.pyfunc.PythonModel):

    @staticmethod
    def _imread(encoding: str) -> np.ndarray:
        img = Image.open(BytesIO(base64.b64decode(encoding.encode())))
        return np.array(img).astype(float)

    def load_context(self, context) -> None:
        pynavio.assert_gpu_available()
        model_path = context.artifacts['model']
        self._session = onnxruntime.InferenceSession(model_path)

    def _predict(self, images: np.ndarray):
        key = self._session.get_inputs()[0].name
        return self._session.run(
            None, {key: images[:, None, :, :].astype(np.float32)})[0]

    @pynavio.prediction_call
    def predict(self, context, model_input):
        imgs = np.stack([*map(self._imread, model_input['image'])])
        return {
            'prediction':
                np.argmax(self._predict(imgs / 255).tolist(), axis=1).tolist()
        }


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, Path]]] = None):

    import torch

    from . import image_pytorch as base

    with TemporaryDirectory() as tmp_dir:
        model_path = f'{tmp_dir}/model.onnx'
        model, trainset = base.train_model()

        dummy_input = torch.randn(10, 1, *base.IMAGE_SHAPE, device=base.DEVICE)
        torch.onnx.export(model,
                          dummy_input,
                          model_path,
                          verbose=True,
                          export_params=True,
                          input_names=['image'],
                          output_names=['prediction'],
                          dynamic_axes={
                              'image': {
                                  0: 'batch_size'
                              },
                              'prediction': {
                                  0: 'batch_size'
                              }
                          })
        example = {
            'image': base.as_base64(Image.fromarray(trainset[0])),
            'digit': str(base._predict(trainset[[0]], model)[0])
        }

        example_request = pynavio.make_example_request(example, 'digit')
        example_request['featureColumns'][0]['type'] = 'image'

        pip_packages = ['mlflow', 'onnxruntime', 'Pillow']

        pynavio.mlflow.to_navio(ImageModel(),
                                example_request=example_request,
                                pip_packages=pip_packages,
                                path=path,
                                code_path=code_path,
                                artifacts={'model': model_path})
