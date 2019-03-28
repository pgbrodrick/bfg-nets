import configparser
import datetime
import os
from typing import List

import keras

from rsCNN.networks import io
from rsCNN.utils import logger


_logger = logger.get_child_logger(__name__)


class HistoryCheckpoint(keras.callbacks.Callback):

    def __init__(self, dir_out, existing_history=None, period=1, verbose=0):
        super().__init__()
        self.dir_out = dir_out
        if existing_history is None:
            existing_history = dict()
        self.existing_history = existing_history
        self.period = period
        self.verbose = verbose
        self.epochs_since_last_save = 0
        self.epoch_begin = None

    def on_train_begin(self, logs=None):
        for key in ('epoch_start', 'epoch_finish'):
            self.existing_history.setdefault(key, list())
        super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        self._save_history()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        # Update times
        epoch_end = datetime.datetime.now()
        self.existing_history['epoch_start'].append(self.epoch_begin)
        self.existing_history['epoch_finish'].append(epoch_end)
        self.epoch_begin = None
        # Save if necessary
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self._save_history()
            self.epochs_since_last_save = 0

    def _save_history(self):
        _logger.debug('Save model history')
        combined_history = io.combine_histories(self.existing_history, self.model.history.history)
        io.save_history(combined_history, self.dir_out)


def get_callbacks(network_config: configparser.ConfigParser, existing_history: dict) -> List[keras.callbacks.Callback]:
    callbacks = [
        HistoryCheckpoint(
            dir_out=network_config['model']['dir_out'],
            existing_history=existing_history,
            period=network_config['callbacks_general']['checkpoint_periods'],
            verbose=network_config['model']['verbosity']
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(network_config['model']['dir_out'], 'model.h5'),
            period=network_config['callbacks_general']['checkpoint_periods'],
            verbose=network_config['model']['verbosity']
        ),
    ]
    if network_config['callbacks_early_stopping']['use_early_stopping']:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=network_config['callbacks_early_stopping']['es_min_delta'],
                patience=network_config['callbacks_early_stopping']['es_patience']
            ),
        )
    if network_config['callbacks_reduced_learning_rate']['use_reduced_learning_rate']:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=network_config['callbacks_reduced_learning_rate']['rlr_factor'],
                min_delta=network_config['callbacks_reduced_learning_rate']['rlr_min_delta'],
                patience=network_config['callbacks_reduced_learning_rate']['rlr_patience']
            ),
        )
    if network_config['callbacks_tensorboard']['use_tensorboard']:
        dir_out = os.path.join(
            network_config['model']['dir_out'],
            network_config['callbacks_tensorboard']['dirname_prefix_tensorboard']
        )
        callbacks.append(
            keras.callbacks.TensorBoard(
                dir_out,
                histogram_freq=network_config['callbacks_tensorboard']['t_histogram_freq'],
                write_graph=network_config['callbacks_tensorboard']['t_write_graph'],
                write_grads=network_config['callbacks_tensorboard']['t_write_grads'],
                write_images=network_config['callbacks_tensorboard']['t_write_images'],
                update_freq=network_config['callbacks_tensorboard']['t_update_freq']
            ),
        )
    if network_config['callbacks_general']['use_terminate_on_nan']:
        callbacks.append(keras.callbacks.TerminateOnNaN())
    return callbacks
