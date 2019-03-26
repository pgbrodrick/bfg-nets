import os
import pickle
from typing import Union

import keras
import matplotlib.pyplot as plt


# TODO:  this script needs to be renamed to io.py or something similar, it's responsible for saving and loading
#  everything associated with models, it'll be referencing by anything that manages model data, like CNN, the future
#  experiment framework, or plotting tools

FILENAME_HISTORY = 'history.pkl'
FILENAME_MODEL = 'model.h5'


def load_history(dir_history: str) -> Union[dict, None]:
    filepath = os.path.join(dir_history, FILENAME_HISTORY)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as file_:
        history = pickle.load(file_)
    return history


def save_history(history: dict, dir_history: str) -> None:
    if not os.path.exists(dir_history):
        os.makedirs(dir_history)
    filepath = os.path.join(dir_history, FILENAME_HISTORY)
    with open(filepath, 'wb') as file_:
        pickle.dump(history, file_)


def load_model(dir_model: str, custom_objects: dict) -> Union[keras.models.Model, None]:
    filepath = os.path.join(dir_model, FILENAME_MODEL)
    if not os.path.exists(filepath):
        return None
    return keras.models.load_model(filepath, custom_objects=custom_objects)


def save_model(model: keras.models.Model, dir_model: str) -> None:
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    filepath = os.path.join(dir_model, FILENAME_MODEL)
    model.save(filepath, overwrite=True)


def combine_histories(existing_history, new_history):
    combined_history = existing_history.copy()
    for key, value in new_history.items():
        combined_history.setdefault(key, list()).extend(value)
    return combined_history

