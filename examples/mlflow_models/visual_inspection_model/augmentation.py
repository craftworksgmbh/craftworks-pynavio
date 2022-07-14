from typing import Optional

import numpy as np
import tensorflow as tf
from albumentations import (Affine, Compose, ElasticTransform, GridDistortion,
                            HorizontalFlip, HueSaturationValue,
                            JpegCompression, RandomBrightnessContrast,
                            RandomContrast, RandomResizedCrop, RGBShift,
                            Rotate)

_transforms = Compose([
    Affine(scale=[0.5, 1.5],
           translate_percent=[0, 0.2],
           shear=[-20, 20],
           rotate=[-45, 45]),
    RandomBrightnessContrast(brightness_limit=[-0.2, 0.4], contrast_limit=0.2),
    JpegCompression(quality_lower=55, quality_upper=100, p=0.5),
    HorizontalFlip(),
    RGBShift(r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1, p=.5),
    GridDistortion(distort_limit=0.1, p=0.3),
    RandomResizedCrop(128, 128, p=0.4)
])


def _augment(image: np.ndarray, mask: np.ndarray, img_size: list) -> tuple:
    aug_data = _transforms(image=image, mask=mask)
    aug_img = tf.image.resize(aug_data["image"], size=img_size)
    aug_mask = tf.image.resize(aug_data["mask"], size=img_size, method="nearest")
    return aug_img, aug_mask


def augment(image: np.ndarray,
            mask: np.ndarray,
            img_size: Optional[list] = None) -> tuple:
    img_size = img_size or [128, 128]
    aug_img, aug_mask = tf.numpy_function(func=_augment,
                                          inp=[image, mask, img_size],
                                          Tout=[tf.float32, tf.float32])
    aug_img.set_shape((*img_size, 3))
    aug_mask.set_shape((*img_size, 1))
    return aug_img, aug_mask
