from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf


def _rglob(path: Path, extensions: set) -> list:
    assert all(sfx == sfx.lower() for sfx in extensions), \
        'Expected all extensions to be lowercase'
    return [
        p.absolute() for p in path.rglob('*') if p.suffix.lower() in extensions
    ]


def _infer_instance_paths(img_paths: list) -> np.ndarray:
    frame = pd.DataFrame({
        'path':
            img_paths,
        'stem': [
            p.stem[:-2] if p.stem.endswith('_l') else p.stem for p in img_paths
        ],
        'is_label': [
            any('label' in part for part in p.parts) for p in img_paths
        ]
    })
    return frame.query('~is_label') \
         .merge(frame.query('is_label'), on='stem') \
         [['path_x', 'path_y']] \
         .astype(str).values.T


def _load_mask(mask_path: str, resize: Optional[tuple] = None) -> tf.Tensor:
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    if resize is not None:
        mask = tf.image.resize(mask, resize)
    return tf.clip_by_value(tf.math.reduce_max(mask, axis=-1, keepdims=True),
                            0, 1)


def _load_image(img_path: str, resize: Optional[tuple] = None) -> tf.Tensor:
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    if resize is not None:
        img = tf.image.resize(img, resize)
    return tf.cast(img, tf.float32) / 255.0


def _as_dataset(img_paths: list,
                mask_paths: list,
                resize: Optional[tuple] = None) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(img_paths, tf.string), tf.cast(mask_paths, tf.string)))
    return dataset.map(lambda img, mask:
                       (_load_image(img, resize), _load_mask(mask, resize)))


def load_dataset(path: str,
                 extensions: set,
                 validation_frac: float = .2,
                 resize: Optional[tuple] = None) -> tuple:
    paths = _infer_instance_paths(_rglob(Path(path), extensions))
    np.random.seed(42)
    is_validation = np.random.rand(paths.shape[-1]) < validation_frac
    np.random.seed()
    train = _as_dataset(*paths[:, ~is_validation], resize)
    valid = _as_dataset(*paths[:, is_validation], resize)
    return train, valid
