import logging
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
