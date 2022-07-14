from functools import partial
from pathlib import Path

import tensorflow as tf

from .augmentation import augment
from .datasets import load_dataset
from .model import setup_model

_BATCH_SIZE = 64
_BUFFER_SIZE = 1000
_INPUT_SHAPE = (224, 224, 3)
_IMG_SIZE = _INPUT_SHAPE[:-1]
_DATA_PATH = './data/visual-inspection/'
_FILE_TYPES = {'.png', '.jpg', '.jpeg'}
_VALIDATION_FRAC = .2
_OUTPUT_CLASSES = 2
_LEARNING_RATE = 1e-4
_EPOCHS = 50


def setup_data() -> tuple:  # keep public for monkeypatching
    path = Path(_DATA_PATH)
    assert path.is_dir(), \
        f'Path {_DATA_PATH} does not exist or is not a directory. ' \
        f'Did you forget to copy the data into {str(path.absolute())}?'

    train, valid = load_dataset(_DATA_PATH,
                                _FILE_TYPES,
                                _VALIDATION_FRAC,
                                resize=_IMG_SIZE)

    train_size, valid_size = len(train), len(valid)

    train = train \
        .cache() \
        .shuffle(_BUFFER_SIZE) \
        .map(partial(augment, img_size=_IMG_SIZE),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(_BATCH_SIZE) \
        .repeat() \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    return train, train_size, valid.batch(_BATCH_SIZE), valid_size


def _setup_callbacks(checkpoint_path: str) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=10,
                                         verbose=0,
                                         mode='min',
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                           save_best_only=True,
                                           monitor='val_loss',
                                           mode='min')
    ]


def train(path: str, epochs: int = _EPOCHS) -> tf.keras.Model:
    checkpoint_path = str(Path(path) / 'checkpoint.h5')
    train, train_size, valid, valid_size = setup_data()
    model = setup_model(_INPUT_SHAPE, _OUTPUT_CLASSES, _LEARNING_RATE)
    model.fit(train,
              epochs=epochs,
              validation_data=valid,
              steps_per_epoch=max(train_size // _BATCH_SIZE, 1),
              validation_steps=max(valid_size // _BATCH_SIZE, 1),
              callbacks=_setup_callbacks(checkpoint_path))
    return model
