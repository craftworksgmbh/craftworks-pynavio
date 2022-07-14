import base64
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pynavio

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SHAPE = (28, 28)


def as_base64(image: Image, rgb: bool = False, fmt: str = 'JPEG') -> str:
    buffered = BytesIO()
    if rgb:
        image.convert('RGB').save(buffered, fmt)
    else:
        image.save(buffered, fmt)
    return base64.b64encode(buffered.getvalue()).decode()


def _make_data():
    kwargs = dict(download=True, transform=transforms.ToTensor())
    return [MNIST('/tmp', train=flag, **kwargs) for flag in [True, False]]


def _make_model(path: str = None):
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1600, 128),  # get in_features arg value from error msg
        nn.ReLU(),
        nn.Linear(128, 10))

    if path is not None:
        model.load_state_dict(torch.load(path))

    return model, nn.CrossEntropyLoss(), optim.Adam(model.parameters())


def train_model():
    BATCH_SIZE = 256

    model, criterion, optimizer = _make_model()
    model.to(DEVICE)

    trainset, testset = _make_data()
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # one epoch only
    model.train()
    for batch in tqdm.tqdm(trainloader):
        X, y = [Variable(x).to(DEVICE) for x in batch]

        # torch accumulates gradients between successive
        # optimizer.step() calls
        optimizer.zero_grad()

        yy = model(X)
        loss = criterion(yy, y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    for batch in tqdm.tqdm(testloader):
        X, y = [x.to(DEVICE) for x in batch]
        yy = model(X)
        correct += (y == yy.argmax(dim=1)).sum().tolist()

    print('Accuracy: ', correct / testset.data.shape[0])
    return model, trainset.data.numpy()


def _predict(images: np.ndarray, model: nn.Sequential) -> np.ndarray:
    return model(torch.Tensor(images[:, None, :, :]).to(DEVICE)) \
        .argmax() \
        .to('cpu') \
        .numpy() \
        .flatten()


class ImageModel(mlflow.pyfunc.PythonModel):

    @staticmethod
    def _imread(encoding: str) -> np.ndarray:
        img = Image.open(BytesIO(base64.b64decode(encoding.encode())))
        return np.array(img).astype(float)

    def load_context(self, context) -> None:
        pynavio.assert_gpu_available()
        self._model, *_ = _make_model(context.artifacts['model'])
        self._model = self._model.to(DEVICE)

    @pynavio.prediction_call
    def predict(self, context, model_input):
        imgs = np.stack([*map(self._imread, model_input['image'])])
        return {'prediction': _predict(imgs / 255, self._model).tolist()}


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          code_path: Optional[List[Union[str, Path]]] = None):

    with TemporaryDirectory() as tmp_dir:
        model_path = f'{tmp_dir}/model.pt'
        model, trainset = train_model()

        torch.save(model.state_dict(), model_path)
        example = {
            'image': as_base64(Image.fromarray(trainset[0])),
            'digit': str(_predict(trainset[[0]], model)[0])
        }

        example_request = pynavio.make_example_request(example, 'digit')
        example_request['featureColumns'][0]['type'] = 'image'

        conda_packages = [
            'cudatoolkit=11.3.1', 'cudnn=8.2.1', 'pytorch=1.10.0',
            'torchaudio=0.10.0', 'torchvision=0.11.0'
        ]

        conda_env = {
            'channels': ['defaults', 'conda-forge', 'pytorch'],
            'dependencies': [
                *conda_packages, {
                    'pip': ['mlflow', 'Pillow', 'tqdm']
                }
            ],
            'name': 'venv'
        }

        pynavio.mlflow.to_navio(ImageModel(),
                                example_request=example_request,
                                conda_env=conda_env,
                                path=path,
                                code_path=code_path,
                                artifacts={'model': model_path},
                                num_gpus=1)
