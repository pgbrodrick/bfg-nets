import os
import pickle
from typing import Union

from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)


FILENAME_HISTORY = 'history.pkl'


def load_history(dir_history: str, filename: str = None) -> Union[dict, None]:
    filepath = os.path.join(dir_history, filename or FILENAME_HISTORY)
    if not os.path.exists(filepath):
        _logger.debug('History not loaded; file does not exist at {}'.format(filepath))
        return None
    _logger.debug('Load history from {}'.format(filepath))
    with open(filepath, 'rb') as file_:
        history = pickle.load(file_)
    return history


def save_history(history: dict, dir_history: str, filename: str = None) -> None:
    if not os.path.exists(dir_history):
        _logger.debug('Create directory to save history at {}'.format(dir_history))
        os.makedirs(dir_history)
    filepath = os.path.join(dir_history, filename or FILENAME_HISTORY)
    _logger.debug('Save history to {}'.format(filepath))
    with open(filepath, 'wb') as file_:
        pickle.dump(history, file_)


def combine_histories(existing_history: dict, new_history: dict) -> dict:
    combined_history = existing_history.copy()
    for key, value in new_history.items():
        combined_history.setdefault(key, list()).extend(value)
    return combined_history
