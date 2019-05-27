import logging
import os
from typing import Union

import keras


_logger = logging.getLogger(__name__)


DEFAULT_FILENAME_MODEL = 'model.h5'


def load_model(filepath: str, custom_objects: dict = None) -> Union[keras.models.Model, None]:
    """Loads model from serialized file.

    Args:
        filepath: Filepath from which model is loaded.
        custom_objects: Custom objects necessary to build model, including loss functions.

    Returns:
        Keras model object if it exists at path.
    """
    if not os.path.exists(filepath):
        _logger.debug('Model not loaded; file does not exist at {}'.format(filepath))
        return None
    _logger.debug('Load model from {}'.format(filepath))
    return keras.models.load_model(filepath, custom_objects=custom_objects)


def save_model(model: keras.models.Model, filepath: str) -> None:
    """Saves model to serialized file.

    Args:
        model: Keras model object.
        filepath: Filepath to which model is saved.

    Returns:
        None.
    """
    if not os.path.exists(os.path.dirname(filepath)):
        _logger.debug('Create directory to save model at {}'.format(os.path.dirname(filepath)))
        os.makedirs(os.path.dirname(filepath))
    _logger.debug('Save model to {}'.format(filepath))
    model.save(filepath, overwrite=True)
