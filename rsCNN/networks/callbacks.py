import datetime
import logging
import os
from typing import List

import keras

from rsCNN.configs import Config
from rsCNN.networks import histories, models


_logger = logging.getLogger(__name__)


_DIR_TENSORBOARD = 'tensorboard'


class HistoryCheckpoint(keras.callbacks.Callback):
    """A custom Keras callback for checkpointing model training history and associated information.
    """
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
        self.existing_history['train_start'] = datetime.datetime.now()
        super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        _logger.debug('Ending network training')
        _logger.debug('on_training_end logs: {}'.format(logs))
        self.existing_history['train_finish'] = datetime.datetime.now()
        self._save_history()

    def on_batch_begin(self, batch, logs=None):
        _logger.debug('Beginning new batch')
        _logger.debug('on_batch_begin logs: {}'.format(logs))

    def on_batch_end(self, batch, logs=None):
        _logger.debug('Ending batch')
        _logger.debug('on_batch_end logs: {}'.format(logs))

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


def get_model_callbacks(config: Config, existing_history: dict) -> List[keras.callbacks.Callback]:
    """Creates model callbacks from a rsCNN config.

    Args:
        config: rsCNN config.
        existing_history: Existing model training history if the model has already been partially or completely trained.

    Returns:
        List of model callbacks.
    """
    callbacks = [
        HistoryCheckpoint(
            dir_out=config.model_training.dir_out,
            existing_history=existing_history,
            period=config.callback_general.checkpoint_periods,
            verbose=config.model_training.verbosity,
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(config.model_training.dir_out, models.FILENAME_MODEL),
            period=config.callback_general.checkpoint_periods,
            verbose=config.model_training.verbosity,
        ),
    ]
    if config.callback_early_stopping.use_callback:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=config.callback_early_stopping.min_delta,
                patience=config.callback_early_stopping.patience,
            ),
        )
    if config.callback_reduced_learning_rate.use_callback:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=config.callback_reduced_learning_rate.factor,
                min_delta=config.callback_reduced_learning_rate.min_delta,
                patience=config.callback_reduced_learning_rate.patience,
            ),
        )
    if config.callback_tensorboard.use_callback:
        dir_out = os.path.join(config.model_training.dir_out, _DIR_TENSORBOARD)
        callbacks.append(
            keras.callbacks.TensorBoard(
                dir_out,
                histogram_freq=config.callback_tensorboard.histogram_freq,
                write_graph=config.callback_tensorboard.write_graph,
                write_grads=config.callback_tensorboard.write_grads,
                write_images=config.callback_tensorboard.write_images,
                update_freq=config.callback_tensorboard.update_freq,
            ),
        )
    if config.callback_general.use_terminate_on_nan:
        callbacks.append(keras.callbacks.TerminateOnNaN())
    return callbacks
