import datetime
import json

import keras

from utils import logger


_logger = logger.get_child_logger(__name__)


class HistoryCheckpoint(keras.callbacks.Callback):

    def __init__(self, filepath, existing_history=None, period=1, verbose=0):
        super().__init__()
        self.filepath = filepath
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
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        history = self._format_history(self.model.history, self.existing_history)
        save_history(history, self.filepath)
        _logger.debug('Epoch %05d: saving history to %s' % (epoch + 1, self.filepath))
        if self.verbose > 0:
            print('\nEpoch %05d: saving history to %s' % (epoch + 1, self.filepath))

    def _format_history(self, new_history, old_history=None):
        if old_history is None:
            old_history = dict()
        combined_history = old_history.copy()
        for key, value in new_history.history.items():
            combined_history.setdefault(key, list()).extend(value)
        return combined_history


# TODO:  these functions will need to be moved to wherever handles history state, not really callback relevant
def load_history(filepath):
    with open(filepath, 'r') as file_:
        history = json.load(file_)
    return history


def save_history(history, filepath):
    with open(filepath, 'wb') as file_:
        json.dump(history, file_)


def get_callbacks(network_config):
    if network_config.append_existing:
        existing_history = load_history(network_config.filepath_history_out)
    else:
        existing_history = dict()
    callbacks = [
        HistoryCheckpoint(
            network_config.filepath_history_out,
            existing_history=existing_history,
            period=network_config.checkpoint_periods,
            verbose=network_config.verbosity
        ),
        keras.callbacks.ModelCheckpoint(
            network_config.filepath_model_out,
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
                network_config.filepath_tensorboard_out,
                histogram_freq=network_config.tensorboard_histogram_freq,
                write_graph=network_config.tensorboard_write_graph,
                write_grads=network_config.tensorboard_write_grads,
                write_images=network_config.tensorboard_write_images,
                update_freq=network_config.tensorboard_update_freq,
            ),
        )
    if network_config.callbacks.use_terminate_on_nan:
        callbacks.append(keras.callbacks.TerminateOnNaN())
    return callbacks
