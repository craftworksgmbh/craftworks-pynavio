from typing import Optional

from numpy import ndarray
import tensorflow as tf
from scipy.special import softmax
from tensorflow.keras import backend as K

_SOFTMAX = tf.keras.layers.Softmax()


def _unet_model(output_channels: int,
                pretrained: bool = True,
                input_shape: Optional[list] = None) -> tf.keras.Model:
    # only used during training, so can be ignored inside mlflow model
    from tensorflow_examples.models.pix2pix import pix2pix
    input_shape = input_shape or [128, 128, 3]

    if pretrained:
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       include_top=False,
                                                       weights="imagenet")
        base_model.trainable = False
    else:
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       include_top=False,
                                                       weights=None)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    base_model_outputs = [
        base_model.get_layer(name).output for name in layer_names
    ]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input,
                                outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(filters=output_channels,
                                           kernel_size=3,
                                           strides=2,
                                           padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def _threshold_mask(pred_mask: tf.Tensor, threshold: float = 0.7) -> tf.Tensor:
    pred_mask = tf.where(_SOFTMAX(pred_mask) > threshold, 1, 0)
    return tf.argmax(pred_mask, axis=-1)


def _create_mask_for_metrics(pred_mask: tf.Tensor) -> tf.Tensor:
    pred_mask = tf.cast(_threshold_mask(pred_mask), tf.float32)
    return pred_mask[..., tf.newaxis]


def _dice_coef(y_true: tf.Tensor,
               y_pred: tf.Tensor,
               smooth: int = 1) -> tf.Tensor:
    y_pred = _create_mask_for_metrics(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def setup_model(input_shape: tuple, output_channels: int,
                learning_rate: float) -> tf.keras.Model:
    model = _unet_model(output_channels=output_channels,
                        input_shape=input_shape)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy', _dice_coef])
    return model


def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        path, custom_objects={'_dice_coef': _dice_coef})


def predict(model: tf.keras.Model, image: ndarray) -> ndarray:
    input_shape = model.layers[0].input.shape[1:3]
    normed_resized = tf.image.resize(image / 255, input_shape)
    logits = model.predict(normed_resized[None, ...])[0]
    return softmax(logits, axis=-1)[..., -1]
