import os
from typing import Union

import keras


FILENAME_MODEL = 'model.h5'


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
