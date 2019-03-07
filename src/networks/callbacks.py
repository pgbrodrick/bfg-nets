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
        with open(self.filepath, 'wb') as file_:
            json.dump(history, file_)
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
