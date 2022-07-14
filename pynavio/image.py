import base64
from io import BytesIO

import numpy as np

_IMAGE = 'PIL.Image'


def _import_image() -> type:
    try:
        from PIL import Image
        return Image
    except ImportError as err:
        raise ImportError('please run "pip install Pillow" to use '
                          'pynavio.image utilities') from err


def imread(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()


def imwrite(path: str, img: np.ndarray):
    Image = _import_image()
    Image.fromarray(img).save(path)


def img_from_b64(encoding: str) -> np.ndarray:
    Image = _import_image()
    img = Image.open(BytesIO(base64.b64decode(encoding.encode())))
    return np.array(img).astype(float)


def img_to_b64(image: _IMAGE, rgb: bool = False, fmt: str = 'JPEG') -> str:
    buffered = BytesIO()
    if rgb:
        image.convert('RGB').save(buffered, fmt)
    else:
        image.save(buffered, fmt)
    return base64.b64encode(buffered.getvalue()).decode()
