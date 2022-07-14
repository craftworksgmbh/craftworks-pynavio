from . import image


def setup(*args, **kwargs):
    kwargs['with_gpu'] = True
    image.setup(*args, **kwargs)
