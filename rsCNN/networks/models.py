import os
from typing import Union

import keras

from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)


FILENAME_MODEL = 'model.h5'


def load_model(dir_model: str, custom_objects: dict = None, filename: str = None) -> Union[keras.models.Model, None]:
    filepath = os.path.join(dir_model, filename or FILENAME_MODEL)
    if not os.path.exists(filepath):
        _logger.debug('Model not loaded; file does not exist at {}'.format(filepath))
        return None
    _logger.debug('Load model from {}'.format(filepath))
    return keras.models.load_model(filepath, custom_objects=custom_objects)


def save_model(model: keras.models.Model, dir_model: str, filename: str = None) -> None:
    if not os.path.exists(dir_model):
        _logger.debug('Create directory to save model at {}'.format(dir_model))
        os.makedirs(dir_model)
    filepath = os.path.join(dir_model, filename or FILENAME_MODEL)
    _logger.debug('Save model to {}'.format(filepath))
    model.save(filepath, overwrite=True)
