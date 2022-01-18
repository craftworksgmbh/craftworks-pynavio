import base64

import numpy as np
from PIL import Image


def imread(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()


def imwrite(path: str, img: np.ndarray):
    Image.fromarray(img).save(path)
