import os
import pickle
from typing import Union


FILENAME_HISTORY = 'history.pkl'


def load_history(dir_history: str, filename: str = None) -> Union[dict, None]:
    filepath = os.path.join(dir_history, filename or FILENAME_HISTORY)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as file_:
        history = pickle.load(file_)
    return history


def save_history(history: dict, dir_history: str, filename: str = None) -> None:
    if not os.path.exists(dir_history):
        os.makedirs(dir_history)
    filepath = os.path.join(dir_history, filename or FILENAME_HISTORY)
    with open(filepath, 'wb') as file_:
        pickle.dump(history, file_)


def combine_histories(existing_history, new_history):
    combined_history = existing_history.copy()
    for key, value in new_history.items():
        combined_history.setdefault(key, list()).extend(value)
    return combined_history
