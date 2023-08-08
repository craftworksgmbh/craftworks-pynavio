import base64
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets.mnist import load_data

import pynavio

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

IMAGE_SHAPE = (28, 28)


def _make_model():
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(*IMAGE_SHAPE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def _write_jpgs(images: np.ndarray, path: str):
    path = Path(path)
    path.mkdir(exist_ok=True)
    for i, image in enumerate(images):
        pynavio.image.imwrite(path / f'img{i + 1}.jpg',
                              image.reshape(*IMAGE_SHAPE))


class ImageModel(mlflow.pyfunc.PythonModel,
                 pynavio.traits.TabularExplainerTraits):
    BG_COLUMN = 'is_background'

    def __init__(self, explanation_format: str):
        super().__init__(explanation_format=explanation_format)

    @staticmethod
    def _imread(encoding: str) -> np.ndarray:
        img = Image.open(BytesIO(base64.b64decode(encoding.encode())))
        img = np.array(img)
        return img.reshape(*img.shape, 1)

    @staticmethod
    def _as_base64(image: Image, rgb: bool = False, fmt: str = 'JPEG') -> str:
        buffered = BytesIO()
        if rgb:
            image.convert('RGB').save(buffered, fmt)
        else:
            image.save(buffered, fmt)
        return base64.b64encode(buffered.getvalue()).decode()

    def _explain(self, images: np.ndarray, background: np.ndarray) -> list:
        import shap
        explainer = shap.GradientExplainer(self._model, background)
        return explainer.shap_values(images)

    def _extract_data(self, images: np.ndarray, data: pd.DataFrame) -> tuple:
        background = data.query(self.BG_COLUMN).index
        _input = data.drop(background).index
        return images[_input], images[background]

    def _draw_image_explanation(self, image: np.ndarray,
                                explanation: np.ndarray) -> str:
        import matplotlib
        import matplotlib.pyplot as plt
        import shap.plots.colors as colors
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()

        max_val = np.nanpercentile(np.abs(explanation), 99.9)
        img = image.squeeze()
        ax.imshow(255 - img,
                  cmap=plt.get_cmap('gray'),
                  alpha=0.15,
                  extent=(-1, *img.shape, -1))

        ax.imshow(explanation.squeeze(),
                  cmap=colors.red_transparent_blue,
                  vmin=-max_val,
                  vmax=max_val)

        ax.set_aspect('equal', 'box')
        ax.axis('off')

        fig.tight_layout()

        canvas.draw()
        width, height = map(int, fig.get_size_inches() * fig.get_dpi())
        result = Image.fromarray(
            np.frombuffer(canvas.buffer_rgba(),
                          dtype='uint8').reshape(height, width, -1))
        return self._as_base64(result, True)

    def _draw_plotly_explanation(self, image: np.ndarray,
                                 explanation: np.ndarray) -> dict:
        import plotly.express as px

        color_scale = pynavio.utils.styling.HEATMAP_COLOR_SCALE

        fig = px.imshow(explanation.squeeze(),
                        color_continuous_scale=color_scale)

        img = Image.fromarray((255 - image.squeeze()).astype('uint8'))
        img = f'data:image/png;base64,{self._as_base64(img, False, "PNG")}'

        background = dict(source=img,
                          xref="x",
                          yref="y",
                          x=0,
                          y=0,
                          sizex=IMAGE_SHAPE[1] - 1,
                          sizey=IMAGE_SHAPE[0] - 1,
                          opacity=0.8,
                          layer="below")

        axis_spec = dict(showgrid=False,
                         showline=False,
                         showticklabels=False,
                         zeroline=False,
                         constrain="domain")

        fig \
            .update_traces(opacity=.8) \
            .add_layout_image(background) \
            .update_xaxes(axis_spec, range=[0, IMAGE_SHAPE[1] - 1]) \
            .update_yaxes(axis_spec, range=[IMAGE_SHAPE[0] - 1, 0],
                          autorange=None) \
            .update_coloraxes(showscale=False)

        return json.loads(fig.to_json())

    def _draw_explanation(self, image: np.ndarray,
                          explanation: np.ndarray) -> Union[dict, str]:
        if self._explanation_format == 'image':
            return self._draw_image_explanation(image, explanation)
        return self._draw_plotly_explanation(image, explanation)

    def load_context(self, context) -> None:
        pynavio.assert_gpu_available()
        from tensorflow.keras.models import load_model
        self._model = load_model(context.artifacts['model'])

    @pynavio.prediction_call
    def predict(self, context, model_input: pd.DataFrame):
        imgs = np.stack([*map(self._imread, model_input['image'])])
        if not self.should_explain(model_input):
            return pd.Series(self._model.predict(imgs / 255).argmax(axis=1)) \
                .pipe(lambda s: {'prediction': s.astype(str).tolist()})

        _input, background = self._extract_data(imgs / 255, model_input)
        if _input.size == 0:
            return dict()

        predictions = self._model.predict(_input).argmax(axis=1)
        shap_values = self._explain(_input, background)
        explanations = [
            shap_values[prediction][index].squeeze()
            for index, prediction in enumerate(predictions)
        ]

        return {
            'prediction':
                predictions.astype(str).tolist(),
            'explanation': [
                self._draw_explanation(255 * im, ex)
                for im, ex in zip(_input, explanations)
            ]
        }


def _read_data(path: str, is_background: bool) -> pd.DataFrame:
    return pd.DataFrame([{
        'image': pynavio.image.imread(p),
        ImageModel.BG_COLUMN: is_background,
        'name': p.stem
    } for p in Path(path).glob('*.jpg')])


def _conda_packages(with_gpu: bool = False) -> Optional[list]:
    if not with_gpu:
        return None
    return ['cudatoolkit=11.3.1', 'cudnn=8.2.1']


def setup(with_data: bool,
          with_oodd: bool,
          explanations: Optional[str] = None,
          path: Optional[str] = None,
          with_gpu: bool = False,
          code_path: Optional[List[Union[str, Path]]] = None):
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_test = map(lambda x: x.reshape(-1, *IMAGE_SHAPE, 1),
                          [x_train, x_test])

    model = _make_model()
    model.fit(x_train / 255,
              np.eye(10)[y_train],
              validation_data=(x_test / 255, np.eye(10)[y_test]),
              batch_size=128,
              epochs=1)

    _write_jpgs(x_test[:10], './images/')
    _write_jpgs(x_train[:100], './background/')

    with TemporaryDirectory() as tmp_dir:
        model_path = f'{tmp_dir}/model.h5'
        model.save(model_path)

        dataset = None
        if with_data:
            df = _read_data('./background/',
                            True).drop(['is_background', 'name'], axis=1)

            df_path = f'{tmp_dir}/imgs.csv'
            df.to_csv(df_path, index=False)
            dataset = dict(name='image-data', path=df_path)

        pynavio.image.imwrite(f'{tmp_dir}/example.jpg',
                              x_test[0].reshape(*IMAGE_SHAPE))
        example = {
            'image': pynavio.image.imread(f'{tmp_dir}/example.jpg'),
            'digit': str(model.predict([x_test[[0]]]).argmax())
        }

        example_request = pynavio.make_example_request(example, 'digit')
        example_request['featureColumns'][0]['type'] = 'image'

        pip_packages = [
            'Pillow', 'tensorflow==2.9.1', 'mlflow==1.15.0', 'protobuf<3.20'
        ]
        if explanations not in [None, 'disabled', 'default']:
            pip_packages.extend(['shap', 'IPython'])
            pip_packages.append({
                'image': 'matplotlib',
                'plotly': 'plotly'
            }[explanations])

        pynavio.mlflow.to_navio(ImageModel(explanations),
                                example_request=example_request,
                                pip_packages=pip_packages,
                                dataset=dataset,
                                path=path,
                                conda_packages=_conda_packages(with_gpu),
                                code_path=code_path,
                                explanations=explanations,
                                artifacts={'model': model_path},
                                oodd='default' if with_oodd else 'disabled',
                                num_gpus=1 if with_gpu else 0)
