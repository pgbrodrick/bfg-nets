import logging
import os

import keras.backend as K
import numpy as np

from bfgn.architectures import config_sections
from bfgn.configuration import configs
from bfgn.data_management.data_core import DataContainer
from bfgn.data_management.sequences import BaseSequence
from bfgn.experiments import callbacks, histories, losses, models
from bfgn.utils import compute_access, logging as root_logging


_logger = logging.getLogger(__name__)

_DEFAULT_FILENAME_LOG = 'log.out'
_KEY_HISTORY_IS_MODEL_TRAINED = 'is_model_trained'


class Experiment(object):
    config = None
    """configs.Config: bfgn configuration object."""
    model = None
    """keras.models.Model: Keras model object."""
    history = None
    """dict: Model training history."""
    loaded_existing_history = None
    """bool: Whether an existing history object was loaded from the model training directory."""
    loaded_existing_model = None
    """bool: Whether an existing model object was loaded from the model training directory."""
    is_model_trained = None
    """bool: Whether model is trained to stopping criteria. If False, either not trained at all or training was stopped 
    before stopping criteria."""
    logger = None
    """logging.Logger: Root logger for Experiment. Available if user wants to directly modify the log formatting,
    handling, or other behavior."""

    def __init__(self, config: configs.Config) -> None:
        errors = config.get_human_readable_config_errors(exclude_sections=['raw_files', 'model_reporting'])
        assert not errors, errors

        self.config = config
        if not os.path.exists(self.config.model_training.dir_out):
            os.makedirs(self.config.model_training.dir_out)

        self.logger = root_logging.get_root_logger(get_log_filepath(self.config))
        self.logger.setLevel(self.config.model_training.log_level)

        # Either save config to new model directory or check that existing config matches provided config
        filepath_config = get_config_filepath(self.config)
        if not os.path.exists(filepath_config):
            configs.save_config_to_file(self.config, filepath_config)
        else:
            config_existing = configs.create_config_from_file(filepath_config)
            config_differences = configs.get_config_differences(self.config, config_existing)
            assert not config_differences, \
                'Provided configuration differs from existing configuration at {}; differing values: {}'.format(
                    filepath_config, config_differences)

    def build_or_load_model(self, data_container: DataContainer = None, num_features: int = None) -> None:
        _logger.info('Building or loading model')

        loss_function = losses.get_cropped_loss_function(
            self.config.model_training.loss_metric,
            2 * self.config.data_build.window_radius,
            2 * self.config.data_build.loss_window_radius,
            self.config.model_training.weighted
        )

        if os.path.exists(get_model_filepath(self.config)):
            _logger.debug('Loading existing model from model training directory at {}'.format(
                get_model_filepath(self.config)))
            self.loaded_existing_model = True
            self.model = models.load_model(
                get_model_filepath(self.config),
                custom_objects={'_cropped_loss': loss_function}
            )
        else:
            _logger.debug('Building new model, no model exists at {}'.format(
                get_model_filepath(self.config)))

            assert data_container or num_features, 'Model building requires either DataContainer or num_features.'
            if not num_features:
                inshape = (
                    self.config.data_build.window_radius * 2,
                    self.config.data_build.window_radius * 2,
                    len(data_container.feature_band_types)
                )
            _logger.debug('Inshape is set to {}, was determined from {}'.format(
                inshape, 'DataContainer' if data_container else 'num_features'
            ))

            self.loaded_existing_model = False
            self.model = config_sections.create_model_from_architecture_config_section(
                self.config.model_training.architecture_name, self.config.architecture, inshape)
            self.model.compile(loss=loss_function, optimizer=self.config.model_training.optimizer)
            self.model.summary()

        if os.path.exists(get_history_filepath(self.config)):
            assert self.loaded_existing_model, \
                'Model training history exists in model training directory, but no model found; directory: {}' \
                .format(self.config.model_training.dir_out)
            _logger.debug('Loading existing history from model training directory at {}'.format(
                get_history_filepath(self.config)))
            self.loaded_existing_history = True
            self.history = histories.load_history(get_history_filepath(self.config))
            if 'lr' in self.history:
                learning_rate = self.history['lr'][-1]
                _logger.debug('Setting learning rate to value from last training epoch: {}'.format(learning_rate))
                K.set_value(self.model.optimizer.lr, learning_rate)
        else:
            assert not self.loaded_existing_model, \
                'Trained model exists in model training directory, but no training history found; directory: {}' \
                .format(self.config.model_training.dir_out)
            self.loaded_existing_history = False
            self.history = {
                'model_name': self.config.model_training.dir_out,
                _KEY_HISTORY_IS_MODEL_TRAINED: False
            }
        self.is_model_trained = self.history[_KEY_HISTORY_IS_MODEL_TRAINED]

    def copy_model_from_experiment(self, reference_config: configs.Config) -> None:
        _logger.info('Loading external model')

        loss_function = losses.get_cropped_loss_function(
            self.config.model_training.loss_metric,
            2 * self.config.data_build.window_radius,
            2 * self.config.data_build.loss_window_radius,
            self.config.model_training.weighted
        )

        if os.path.exists(get_model_filepath(reference_config)):
            _logger.debug('Loading existing model from model training directory at {}'.format(
                get_model_filepath(reference_config)))
            self.loaded_existing_model = True
            self.model = models.load_model(
                get_model_filepath(reference_config),
                custom_objects={'_cropped_loss': loss_function}
            )
        else:
            _logger.error('No model exists at {}'.format(
                get_model_filepath(reference_config)))

        if os.path.exists(get_history_filepath(reference_config)):
            self.loaded_existing_history = True
            self.history = histories.load_history(get_history_filepath(reference_config))
            self.history['model_name'] = self.config.model_training.dir_out
            self.history[_KEY_HISTORY_IS_MODEL_TRAINED] = False
            if 'lr' in self.history:
                learning_rate = self.history['lr'][-1]
                _logger.debug('Setting learning rate to value from last training epoch: {}'.format(learning_rate))
                K.set_value(self.model.optimizer.lr, learning_rate)

        else:
            _logger.warning('No history file found in directory: {}'.format(reference_config.model_training.dir_out))
            self.history = {
                'model_name': self.config.model_training.dir_out,
                _KEY_HISTORY_IS_MODEL_TRAINED: False
            }
        self.is_model_trained = False

    def fit_model_with_data_container(self, data_container: DataContainer, resume_training: bool = False) -> None:
        return self.fit_model_with_sequences(
            data_container.training_sequence, data_container.validation_sequence, resume_training
        )

    def fit_model_with_sequences(
            self,
            training_sequence: BaseSequence,
            validation_sequence: BaseSequence = None,
            resume_training: bool = False
    ) -> None:
        if self.loaded_existing_model:
            assert resume_training, 'Resume must be true to continue training an existing model'
        if self.config.model_training.assert_gpu:
            compute_access.assert_gpu_available()

        model_callbacks = callbacks.get_model_callbacks(self.config, self.history)

        new_history = self.model.fit_generator(
            training_sequence,
            epochs=self.config.model_training.max_epochs,
            verbose=self.config.model_training.verbosity,
            callbacks=model_callbacks,
            validation_data=validation_sequence,
            max_queue_size=2,
            shuffle=False,
            initial_epoch=len(self.history.get('lr', list())),
        )
        self.history = histories.combine_histories(self.history, new_history.history)
        self.history[_KEY_HISTORY_IS_MODEL_TRAINED] = True
        self.is_model_trained = True
        histories.save_history(self.history, get_history_filepath(self.config))
        models.save_model(self.model, get_model_filepath(self.config))

    def calculate_model_memory_footprint(self, batch_size: int) -> float:
        """Calculate model memory footprint. Shamelessly copied from (but not tested rigorously):
        https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model.
        Args:
            batch_size:  Batch size for training.

        Returns:
            Model memory footprint in gigabytes.
        """
        shapes_mem_count = 0
        for l in self.model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(self.model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)])

        if K.floatx() == 'float16':
            number_size = 2.0
        elif K.floatx() == 'float32':
            number_size = 4.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes


def load_experiment_from_directory(dir_experiment: str) -> Experiment:
    filepath_config = os.path.join(dir_experiment, configs.DEFAULT_FILENAME_CONFIG)
    assert os.path.exists(filepath_config), 'Experiment directory must contain a config file.'
    config = configs.create_config_from_file(filepath_config)
    config.model_training.dir_out = dir_experiment
    return Experiment(config)


def get_config_filepath(config: configs.Config) -> str:
    """Get the default config path for experiments.

    Args:
        config: bfgn config object.

    Returns:
        Filepath to model training config.
    """
    return os.path.join(config.model_training.dir_out, configs.DEFAULT_FILENAME_CONFIG)


def get_history_filepath(config: configs.Config) -> str:
    """Get the default model training history path for experiments.

    Args:
        config: bfgn config object.

    Returns:
        Filepath to model training history.
    """
    return os.path.join(config.model_training.dir_out, histories.DEFAULT_FILENAME_HISTORY)


def get_log_filepath(config: configs.Config) -> str:
    """Get the default model training log path for experiments.

    Args:
        config: bfgn config object.

    Returns:
        Filepath to model training log.
    """
    return os.path.join(config.model_training.dir_out, _DEFAULT_FILENAME_LOG)


def get_model_filepath(config: configs.Config) -> str:
    """Get the default model path for experiments.

    Args:
        config: bfgn config object.

    Returns:
        Filepath to model.
    """
    return os.path.join(config.model_training.dir_out, models.DEFAULT_FILENAME_MODEL)
