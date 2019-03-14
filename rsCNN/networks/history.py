import datetime
import os
import pickle

import keras


# TODO:  this script needs to be renamed to io.py or something similar, it's responsible for saving and loading
#  everything associated with models, it'll be referencing by anything that manages model data, like CNN, the future
#  experiment framework, or plotting tools

DATETIME_FORMAT = '%Y%m%d_%H%M%S'

FILENAME_HISTORY = '{}_history.pkl'
FILENAME_MODEL = '{}_model.h5'


def load_history(dir_history: str):
    filename = _get_existing_filename(dir_history, FILENAME_HISTORY)
    if not filename:
        return None
    filepath = os.path.join(dir_history, filename)
    with open(filepath, 'rb') as file_:
        history = pickle.load(file_)
    return history


def save_history(history: dict, dir_history: str):
    filename = _get_existing_filename(dir_history, FILENAME_HISTORY)
    if not filename:
        filename = _set_filename(FILENAME_HISTORY)
    filepath = os.path.join(dir_history, filename)
    with open(filepath, 'wb') as file_:
        pickle.dump(history, file_)


def load_model(dir_model: str, custom_objects: dict):
    filename = _get_existing_filename(dir_model, FILENAME_MODEL)
    if not filename:
        return None
    filepath = os.path.join(dir_model, filename)
    return keras.models.load_model(filepath, custom_objects=custom_objects)


def save_model(model: keras.models.Model, dir_model: str):
    filename = _get_existing_filename(dir_model, FILENAME_MODEL)
    if not filename:
        filename = _set_filename(FILENAME_MODEL)
    filepath = os.path.join(dir_model, filename)
    model.save(filepath, overwrite=True)


# TODO:  we really want the callbacks to handle this based on the CNN __init__ datetime, so this needs to be migrated
#  I'm leaving it here because I'm pooped and rushing
def _set_filename(filename_template):
    now = datetime.datetime.now().strftime(DATETIME_FORMAT)
    return filename_template.format(now)


def _get_existing_filename(dir_, filename_template):
    existing_files = os.listdir(dir_)
    filename_pattern = filename_template.format('')
    filename_match = [filename for filename in existing_files if bool(filename.find(filename_pattern))]
    assert len(filename_match) <= 1, 'Multiple matching filenames found: {}'.format(', '.join(filename_match))
    if filename_match:
        return filename_match[0]
    return None
