import logging
import os
from typing import Callable, Tuple

import keras.backend as K
import numpy as np

from bfgn.architectures import config_sections
from bfgn.configuration import configs
from bfgn.data_management.data_core import DataContainer
from bfgn.data_management.sequences import BaseSequence
from bfgn.experiments import callbacks, histories, losses, models
from bfgn.utils import compute_access
from bfgn.utils import logging as root_logging

_logger = logging.getLogger(__name__)

_DEFAULT_FILENAME_LOG = "log.out"
KEY_HISTORY_IS_MODEL_TRAINED = "is_model_trained"


class Experiment(object):
    config = None
    """configs.Config: bfgn configuration object."""
    filepath_config = None
    """str: Filepath to which model config is saved."""
    model = None
    """keras.models.Model: Keras model object."""
    model_gbs = None
    """float: Estimated size of model in gigabytes. Please note that this is just an estimate and that it may be
    incorrect. We're getting unusual results with small models causing OOM errors and need to determine whether this is
    due to model size estimates being too low, configuration issues with the compute resources including GPUs, or 
    other."""
    filepath_model = None
    """str: Filepath to which Keras model is saved."""
    history = None
    """dict: Model training history."""
    filepath_history = None
    """str: Filepath to which model history is saved."""
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
    filepath_logs = None
    """str: Filepath to which model logs are saved."""
    _init_epoch = 0
    """int: Initial epoch for model training"""

    def __init__(self, config: configs.Config) -> None:
        errors = config.get_human_readable_config_errors(exclude_sections=["raw_files", "model_reporting"])
        assert not errors, errors
        self.config = config
        self.filepath_config = get_config_filepath(config)
        self.filepath_model = get_model_filepath(config)
        self.filepath_history = get_history_filepath(config)
        self.filepath_logs = get_log_filepath(config)

        if not os.path.exists(self.config.model_training.dir_out):
            os.makedirs(self.config.model_training.dir_out)

        self.logger = root_logging.get_root_logger(self.filepath_logs)
        self.logger.setLevel(self.config.model_training.log_level)

        self._save_new_config_or_assert_existing_config_matches()

    def _save_new_config_or_assert_existing_config_matches(self) -> None:
        if not os.path.exists(self.filepath_config):
            configs.save_config_to_file(self.config, self.filepath_config)
        else:
            config_existing = configs.create_config_from_file(self.filepath_config)
            config_differences = configs.get_config_differences(self.config, config_existing)
            assert (
                not config_differences
            ), "Provided configuration differs from existing configuration at {}; differing values: {}".format(
                self.filepath_config, config_differences
            )

    def build_or_load_model(self, data_container: DataContainer = None, num_features: int = None) -> None:
        assert bool(data_container) != bool(
            num_features
        ), "Model building requires either DataContainer or num_features, not both."
        if data_container:
            num_features = len(data_container.feature_band_types)
        input_shape = (self.config.data_build.window_radius * 2, self.config.data_build.window_radius * 2, num_features)
        if not os.path.exists(self.filepath_model):
            self._build_new_model(input_shape)
            self._build_new_history()
        else:
            self._load_existing_model(input_shape)
            self._load_existing_history()

    def _build_new_model(self, input_shape: Tuple[int, int, int]) -> None:
        self.model = config_sections.create_model_from_architecture_config_section(
            self.config.model_training.architecture_name, self.config.architecture, input_shape
        )
        self.model.compile(loss=self._create_loss_function(), optimizer=self.config.model_training.optimizer)
        self.model_gbs = self.calculate_model_memory_footprint(self.config.data_samples.batch_size)
        self.logger.debug("Estimated model size: {} GBs".format(self.model_gbs))
        models.save_model(self.model, self.filepath_model)
        self.loaded_existing_model = False

    def _build_new_history(self) -> None:
        self.history = {"model_name": self.config.model_training.dir_out, KEY_HISTORY_IS_MODEL_TRAINED: False}
        histories.save_history(self.history, self.filepath_history)
        self.is_model_trained = False
        self.loaded_existing_history = False

    def _load_existing_model(self, input_shape: Tuple[int, int, int]) -> None:
        self.model = models.load_model(
            self.filepath_model, custom_objects={"_cropped_loss": self._create_loss_function()}
        )
        self.model_gbs = self.calculate_model_memory_footprint(self.config.data_samples.batch_size)
        self.logger.debug("Estimated model size: {} GBs".format(self.model_gbs))
        existing_shape = self.model.layers[0].input_shape[1:]
        assert (
            existing_shape == input_shape
        ), "Existing model's input shape ({}) does not match provided input shape ({})".format(
            existing_shape, input_shape
        )
        self.loaded_existing_model = True

    def _load_existing_history(self):
        # TODO:  Phil, confirm this is what you want/expect
        self.history = histories.load_history(get_history_filepath(self.config))
        self.is_model_trained = self.history[KEY_HISTORY_IS_MODEL_TRAINED]
        self.loaded_existing_history = True
        _logger.debug(
            "Setting initial epoch and learning rate from best model epoch, if possible, otherwise from "
            + "most recent model epoch"
        )
        if "val_loss" not in self.history and "lr" not in self.history:
            return
        if "val_loss" in self.history:
            self._init_epoch = 1 + np.argmin(self.history["val_loss"])
        elif "lr" in self.history:
            self._init_epoch = len(self.history["lr"])
        init_learning_rate = self.history["lr"][self._init_epoch - 1]
        K.set_value(self.model.optimizer.lr, init_learning_rate)

    def _create_loss_function(self) -> Callable:
        return losses.get_cropped_loss_function(
            self.config.model_training.loss_metric,
            2 * self.config.data_build.window_radius,
            2 * self.config.data_build.loss_window_radius,
            self.config.model_training.weighted,
        )

    def fit_model_with_data_container(self, data_container: DataContainer, resume_training: bool = False) -> None:
        return self.fit_model_with_sequences(
            data_container.training_sequence, data_container.validation_sequence, resume_training
        )

    def fit_model_with_sequences(
        self, training_sequence: BaseSequence, validation_sequence: BaseSequence = None, resume_training: bool = False
    ) -> None:
        if self.loaded_existing_model:
            assert resume_training, "Resume must be true to continue training an existing model"
        if self.config.model_training.assert_gpu:
            compute_access.assert_gpu_available()

        model_callbacks = callbacks.get_model_callbacks(self.config, self.history)

        new_history = self.model.fit_generator(
            training_sequence,
            epochs=self.config.model_training.max_epochs,
            verbose=self.config.model_training.verbosity,
            callbacks=model_callbacks,
            validation_data=validation_sequence,
            max_queue_size=min(10, 2 * compute_access.get_count_available_cpus()),
            use_multiprocessing=False,
            shuffle=False,
            initial_epoch=self._init_epoch,
        )
        models.save_model(self.model, get_model_filepath(self.config))
        self.is_model_trained = True
        self.history = histories.combine_histories(self.history, new_history.history)
        self.history[KEY_HISTORY_IS_MODEL_TRAINED] = True
        histories.save_history(self.history, get_history_filepath(self.config))

    def calculate_model_memory_footprint(self, batch_size: int) -> float:
        """Calculate model memory footprint. Shamelessly copied from (but not tested rigorously):
        https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model.
        Please note that this is only an estimate and may be incorrect, but we have not been able to test this further.
        Args:
            batch_size:  Batch size for training.

        Returns:
            Model memory footprint in gigabytes.
        """
        shapes_mem_count = 0
        for l in self.model.layers:
            if l.name.startswith("concatenate"):
                continue
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(self.model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)])

        if K.floatx() == "float16":
            number_size = 2.0
        elif K.floatx() == "float32":
            number_size = 4.0
        if K.floatx() == "float64":
            number_size = 8.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes


def load_experiment_from_directory(dir_experiment: str) -> Experiment:
    filepath_config = os.path.join(dir_experiment, configs.DEFAULT_FILENAME_CONFIG)
    assert os.path.exists(filepath_config), "Experiment directory must contain a config file."
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
