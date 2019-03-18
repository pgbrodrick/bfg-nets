import datetime
import os
import pickle

import keras
import matplotlib.pyplot as plt


# TODO:  this script needs to be renamed to io.py or something similar, it's responsible for saving and loading
#  everything associated with models, it'll be referencing by anything that manages model data, like CNN, the future
#  experiment framework, or plotting tools

DATETIME_FORMAT = '%Y%m%d_%H%M%S'

FILENAME_HISTORY = '{}_history.pkl'
FILENAME_MODEL = '{}_model.h5'


def load_history(dir_history: str) -> dict:
    filename = _get_existing_filename(dir_history, FILENAME_HISTORY)
    if not filename:
        return None
    filepath = os.path.join(dir_history, filename)
    with open(filepath, 'rb') as file_:
        history = pickle.load(file_)
    return history


def save_history(history: dict, dir_history: str) -> None:
    filename = _get_existing_filename(dir_history, FILENAME_HISTORY)
    if not filename:
        filename = _set_filename(FILENAME_HISTORY)
    filepath = os.path.join(dir_history, filename)
    with open(filepath, 'wb') as file_:
        pickle.dump(history, file_)


def load_model(dir_model: str, custom_objects: dict) -> keras.models.Model:
    filename = _get_existing_filename(dir_model, FILENAME_MODEL)
    if not filename:
        return None
    filepath = os.path.join(dir_model, filename)
    return keras.models.load_model(filepath, custom_objects=custom_objects)


def save_model(model: keras.models.Model, dir_model: str) -> None:
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


def plot_history(history, path_out=None):
    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
    # Epoch times and delays
    ax = axes.ravel()[0]
    epoch_time = [(finish - start).seconds for start, finish in zip(history['epoch_start'], history['epoch_finish'])]
    epoch_delay = [(start - finish).seconds for start, finish
                   in zip(history['epoch_start'][1:], history['epoch_finish'][:-1])]
    ax.plot(epoch_time, c='black', label='Epoch time')
    ax.plot(epoch_delay, '--', c='blue', label='Epoch delay')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Seconds')
    ax.legend()
    # Epoch times different view
    ax = axes.ravel()[1]
    dts = [epoch.strftime('%d %H:%M') for epoch in history['epoch_finish']]
    ax.hist(dts)
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylabel('Epochs completed')
    # Loss
    ax = axes.ravel()[2]
    ax.plot(history['loss'][-160:], c='black', label='Training loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'][-160:], '--', c='blue', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    # Learning rate
    ax = axes.ravel()[3]
    ax.plot(history['lr'][-160:], c='black')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')
    if path_out:
        fig.savefig(path_out)
    else:
        fig.show()
