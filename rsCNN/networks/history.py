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
