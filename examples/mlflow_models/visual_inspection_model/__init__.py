import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from PIL import Image
from plotly.graph_objects import Figure
from scipy import ndimage

import pynavio

from .model import load_model, predict

_THRESHOLD = .2
_COLOR_SCALE = ['rgba(30,136,229,0)', 'rgba(255,13,87,255)']


def _get_example_image() -> str:
    path = Path(__file__).parent / 'example.png'
    return pynavio.image.imread(str(path))


def _fake_data(path: str) -> None:
    """ Required to enable explanations """
    pd.DataFrame({'image': list('abc'), 'knots': [1, 2, 3]}) \
      .to_csv(path, index=False)


def _close_open(image: np.ndarray) -> np.ndarray:
    grid = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    element = (np.dstack(grid)**2).sum(axis=-1) <= 4
    closed = ndimage.binary_closing(image, element)
    return ndimage.binary_opening(image, element).astype(np.uint8)


def _prepare_image(image: np.ndarray) -> Image:
    # we want to limit image resolution here for performance reasons
    # plotly is very slow for high resolution heatmaps
    target = (256, 256)
    actual = image.shape[:2]
    img = Image.fromarray(image.astype(np.uint8))
    if any(dim > desired for dim, desired in zip(actual, target)):
        img.thumbnail(target)  # resize, preserving aspect ratio
    return img


def _count_components(image: np.ndarray) -> int:
    return ndimage.label(image)[1]


def _draw_plotly_explanation(image: Image, explanation: np.ndarray) -> Figure:
    shape = (image.height, image.width)
    fig = px.imshow(explanation.squeeze(), color_continuous_scale=_COLOR_SCALE)
    img_b64 = pynavio.image.img_to_b64(image, False, "PNG")

    background = dict(source=f'data:image/png;base64,{img_b64}',
                      xref="x",
                      yref="y",
                      x=0,
                      y=0,
                      sizex=shape[1] - 1,
                      sizey=shape[0] - 1,
                      opacity=.8,
                      layer="below")

    axis_spec = dict(showgrid=False,
                     showline=False,
                     showticklabels=False,
                     zeroline=False,
                     constrain="domain")

    return fig \
        .update_traces(opacity=.8) \
        .add_layout_image(background) \
        .update_xaxes(axis_spec, range=[0, shape[1] - 1]) \
        .update_yaxes(axis_spec, range=[shape[0] - 1, 0], autorange=None) \
        .update_coloraxes(showscale=False, cmin=0, cmax=1)


class KnotDetector(mlflow.pyfunc.PythonModel):

    BG_COLUMN = 'is_background'

    def __init__(self, model: tf.keras.Model):
        self._model = model

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self._model = load_model(context.artifacts['model'])

    def _detect(self, image: np.ndarray) -> tuple:
        prediction = (predict(self._model, image) * 255).astype(np.uint8)
        prediction = Image.fromarray(prediction)
        img = _prepare_image(image)
        shape = (img.width, img.height)
        return np.array(prediction.resize(shape)) / 255, img

    def _predict(self, image: np.ndarray) -> tuple:
        detection, img = self._detect(image)
        binary = _close_open((detection > _THRESHOLD).astype(np.uint8))
        return _count_components(binary), detection, img

    @pynavio.prediction_call
    def predict(self, context: mlflow.pyfunc.PythonModelContext,
                model_input: pd.DataFrame) -> dict:
        if self.BG_COLUMN in model_input.columns:
            # ignore background data, because it's fake and not needed
            model_input = model_input.query('~is_background')
        images = model_input.image.map(pynavio.image.img_from_b64)
        predictions = [*map(self._predict, images)]
        return {
            'prediction': [count for count, *_ in predictions],
            'explanation': [
                json.loads(_draw_plotly_explanation(image, mask).to_json())
                for _, mask, image in predictions
            ]
        }


def mock_data():
    # expects batches of shape _INPUT_SHAPE
    feature = np.random.rand(*train._INPUT_SHAPE)[None, ...]

    # expects batches of shape (_INPUT_SHAPE[0], _INPUT_SHAPE[1], 1)
    label = np.where(feature[..., 0] > .5, 1, 0)[..., None]

    def _as_generator(feature, label):
        while True:
            yield feature, label

    generator = _as_generator(feature.repeat(train._BATCH_SIZE, 0),
                              label.repeat(train._BATCH_SIZE, 0))

    return generator, 1, generator, 1


def setup(*args, **kwargs):
    # import training logic here to avoid having to add training
    # dependencies, i.e. albumentations and tensorflow_examples
    from .train import train

    with TemporaryDirectory() as tmp_dir:
        model = train(tmp_dir)
        model_path = f'{tmp_dir}/model.h5'
        model.save(model_path)

        detector = KnotDetector(model)
        frame = pd.DataFrame({'image': [_get_example_image()]})
        frame['knots'] = detector.predict(None, frame)['prediction']
        schema = pynavio.make_example_request(frame, 'knots')
        schema['featureColumns'][0]['type'] = 'image'

        detector._model = None  # do not save via pickle

        conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'python=3.9.12', 'pip=22.0.4', {
                    'pip': [
                        'Pillow==9.3.0', 'plotly==5.9.0', 'scipy==1.11.4', 'tensorflow==2.11.1',
                        'mlflow==2.9.2', 'protobuf<3.20', 'pynavio==0.2.4'
                    ]
                }
            ],
            'name': 'venv'
        }

        data_path = f'{tmp_dir}/knot-detections.csv'
        _fake_data(data_path)

        pynavio.mlflow.to_navio(detector,
                                example_request=schema,
                                explanations='plotly',
                                artifacts={'model': model_path},
                                path=kwargs.get('path'),
                                conda_env=conda_env,
                                dataset=dict(name='knot-detections',
                                             path=data_path),
                                code_path=kwargs.get('code_path'),
                                oodd='disabled')
