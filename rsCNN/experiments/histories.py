import logging
import os
import pickle
from typing import Union


_logger = logging.getLogger(__name__)


DEFAULT_FILENAME_HISTORY = 'model_history.pkl'


def load_history(filepath: str) -> Union[dict, None]:
    """Loads model training history from serialized file.

    Args:
        filepath: Filepath from which history is loaded.

    Returns:
        History object if it exists at path.
    """
    if not os.path.exists(filepath):
        _logger.debug('History not loaded; file does not exist at {}'.format(filepath))
        return None
    _logger.debug('Load history from {}'.format(filepath))
    with open(filepath, 'rb') as file_:
        history = pickle.load(file_)
    return history


def save_history(history: dict, filepath: str) -> None:
    """Saves model training history to serialized file

    Args:
        history: Model training history object.
        filepath: Filepath to which history is saved.

    Returns:
        None.
    """
    if not os.path.exists(os.path.dirname(filepath)):
        _logger.debug('Create directory to save history at {}'.format(os.path.dirname(filepath)))
        os.makedirs(os.path.dirname(filepath))
    _logger.debug('Save history to {}'.format(filepath))
    with open(filepath, 'wb') as file_:
        pickle.dump(history, file_)


def combine_histories(existing_history: dict, new_history: dict) -> dict:
    """Combines model training history objects, such that the new history from a more recent training session is
    appended to the existing history from a previous training session.

    Args:
        existing_history: Model training history from a previous training session.
        new_history: Model training history from a more recent training session.

    Returns:
        Combined model training history.
    """
    combined_history = existing_history.copy()
    for key, value in new_history.items():
        combined_history.setdefault(key, list()).extend(value)
    return combined_history
