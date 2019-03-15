import datetime
import os
from typing import List

import keras

from rsCNN.networks import history
from rsCNN.networks.network_config import NetworkConfig
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
        combined_history = self.existing_history.copy()
        for key, value in self.model.history.history.items():
            combined_history.setdefault(key, list()).extend(value)
        history.save_history(combined_history, dir_out)


def get_callbacks(network_config: NetworkConfig, existing_history: dict) -> List[keras.callbacks.Callback]:
    # TODO:  remove the hacky implementation of modelcheckpoint filepath, needs to use datetime from history module
    callbacks = [
        HistoryCheckpoint(
            dir_out=network_config.dir_out,
            existing_history=existing_history,
            period=network_config.checkpoint_periods,
            verbose=network_config.verbosity
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(network_config.dir_out, 'model.h5'),
            period=network_config.checkpoint_periods,
            verbose=network_config.verbosity
        ),
    ]
    if network_config.callbacks_use_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=network_config.early_stopping_min_delta,
                patience=network_config.early_stopping_patience
            ),
        )
    if network_config.callbacks_use_reduced_learning_rate:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=network_config.reduced_learning_rate_factor,
                min_delta=network_config.reduced_learning_rate_min_delta,
                patience=network_config.reduced_learning_rate_patience
            ),
        )
    if network_config.callbacks_use_tensorboard:
        callbacks.append(
            keras.callbacks.TensorBoard(
                network_config.dirname_prefix_tensorboard,
                histogram_freq=network_config.tensorboard_histogram_freq,
                write_graph=network_config.tensorboard_write_graph,
                write_grads=network_config.tensorboard_write_grads,
                write_images=network_config.tensorboard_write_images,
                update_freq=network_config.tensorboard_update_freq
            ),
        )
    if network_config.callbacks_use_terminate_on_nan:
        callbacks.append(keras.callbacks.TerminateOnNaN())
    return callbacks
