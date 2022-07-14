import logging
import os
import traceback
from functools import wraps


def prediction_call(predict_fn: callable) -> callable:
    logger = logging.getLogger('gunicorn.error')

    @wraps(predict_fn)
    def wrapper(*args, **kwargs) -> dict:
        try:
            return predict_fn(*args, **kwargs)
        except Exception as exc:
            logger.exception('Prediction call failed')
            return {
                'error_code': exc.__class__.__name__,
                'message': str(exc),
                'stack_trace': traceback.format_exc()
            }

    return wrapper


def assert_gpu_available() -> None:
    """ Helper for ensuring GPU models actually register the GPU

    Only triggers an error if NVIDIA_VISIBLE_DEVICES env. var. is defined
    """
    key = 'NVIDIA_VISIBLE_DEVICES'
    if key not in os.environ:
        return

    ok = False
    logger = logging.getLogger('gunicorn.error')
    logger.info("Checking GPU availability: %s=%s", key, os.environ[key])
    try:
        import tensorflow as tf
        gpu_devices = tf.config.list_physical_devices('GPU')
        logger.info('tensorflow found gpu devices: %s', gpu_devices)
        assert gpu_devices, 'Empty GPU device list from tensorflow'
        ok = True
    except ImportError:
        logger.info('Ignoring tensorflow check: module not found')
    try:
        import torch
        assert torch.cuda.is_available(), 'Torch does not register GPU'
        ok = True
    except ImportError:
        logger.info('Ignoring pytorch check: module not found')
    try:
        import onnxruntime
        assert onnxruntime.get_device() == 'GPU'
        ok = True
    except ImportError:
        logger.info('Ignoring onnxruntime check: module not found')
    assert ok, \
        'Neither tensorflow, onnxruntime nor pytorch are installed. ' \
        'Are you sure this is a GPU-ready model?'
