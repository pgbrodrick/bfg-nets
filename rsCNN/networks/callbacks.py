import configparser
import datetime
import os
from typing import List

import keras

from rsCNN.networks import histories
from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)


class HistoryCheckpoint(keras.callbacks.Callback):
    # TODO:  review logged logs from the below methods to determine what should be logged

    def __init__(self, dir_out, existing_history=None, period=1, verbose=0):
        super().__init__()
        # TODO:  remove this log line
        if hasattr(self, 'model'):
            _logger.debug('init model attributes and methods:  {}'.format(dir(self.model)))
            if hasattr(self.model, 'model'):
                _logger.debug('init model.model attributes and methods:  {}'.format(dir(self.model.model)))
        self.dir_out = dir_out
        if existing_history is None:
            existing_history = dict()
        self.existing_history = existing_history
        self.period = period
        self.verbose = verbose
        self.epochs_since_last_save = 0
        self.epoch_begin = None

    def on_train_begin(self, logs=None):
        # TODO:  remove this log line
        if hasattr(self, 'model'):
            _logger.debug('train begin model attributes and methods:  {}'.format(dir(self.model)))
            if hasattr(self.model, 'model'):
                _logger.debug('train begin model.model attributes and methods:  {}'.format(dir(self.model.model)))
        _logger.debug('Beginning network training')
        _logger.debug('on_training_begin logs: {}'.format(logs))
        for key in ('epoch_start', 'epoch_finish'):
            self.existing_history.setdefault(key, list())
        super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        _logger.debug('Ending network training')
        _logger.debug('on_training_end logs: {}'.format(logs))
        self._save_history()

    def on_epoch_begin(self, epoch, logs=None):
        _logger.debug('Beginning new epoch')
        _logger.debug('on_epoch_begin logs: {}'.format(logs))
        self.epoch_begin = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        _logger.debug('Ending epoch')
        _logger.debug('on_epoch_end logs: {}'.format(logs))
        # Update times
        epoch_end = datetime.datetime.now()
        self.existing_history['epoch_start'].append(self.epoch_begin)
        self.existing_history['epoch_finish'].append(epoch_end)
        self.epoch_begin = None
        # Save if necessary
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            _logger.debug('Checkpointing')
            self._save_history()
            self.epochs_since_last_save = 0

    def _save_history(self):
        _logger.debug('Save model history')
        if hasattr(self.model, 'history'):
            new_history = self.model.history.history
        elif hasattr(self.model, 'model'):
            assert hasattr(self.model.model, 'history'), \
                'Parallel models are doing something unusual with histories. Tell Nick and let\'s debug.'
            new_history = self.model.model.history
        combined_history = histories.combine_histories(self.existing_history, new_history)
        histories.save_history(combined_history, self.dir_out)


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
