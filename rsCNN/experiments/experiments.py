import logging
import os

import keras.backend as K
import numpy as np

from rsCNN.architectures import config_sections
from rsCNN.configuration import configs
from rsCNN.data_management.training_data import Dataset
from rsCNN.data_management.sequences import BaseSequence
from rsCNN.experiments import callbacks, histories, losses, models
from rsCNN.utils import gpus


_logger = logging.getLogger(__name__)


# TODO:  document this file

class Experiment(object):
    config = None
    """configs.Config: rsCNN configuration object."""
    model = None
    """keras.models.Model: Keras model object."""
    history = None
    """dict: Model training history."""
    loaded_existing_history = None
    """bool: Whether an existing history object was loaded from the model training directory."""
    loaded_existing_model = None
    """bool: Whether an existing model object was loaded from the model training directory."""

    def __init__(self, config: configs.Config) -> None:
        self.config = config
        if not os.path.exists(self.config.model_training.dir_out):
            os.makedirs(self.config.model_training.dir_out)
        if os.path.exists(get_config_filepath(self.config.model_training.dir_out)):
            config_existing = configs.create_config_from_file(get_config_filepath(self.config.model_training.dir_out))
            config_differences = configs.get_config_differences(self.config, config_existing)
            assert not config_differences, \
                'Provided configuration differs from existing configuration at {}; differing values: {}'.format(
                    get_config_filepath(self.config.model_training.dir_out), config_differences)
        else:
            configs.save_config_to_file(self.config, get_config_filepath(self.config.model_training.dir_out))

    def build_or_load_model(self):
        _logger.info('Building or loading model')
        loss_function = losses.cropped_loss(
            self.config.model_training.loss_metric,
            self.config.architecture.inshape[0],
            2 * self.config.data_build.loss_window_radius,
            self.config.model_training.weighted
        )
        if os.path.exists(get_model_filepath(self.config.model_training.dir_out)):
            _logger.debug('Loading existing model from model training directory at {}'.format(
                get_model_filepath(self.config.model_training.dir_out)))
            self.loaded_existing_model = True
            self.model = models.load_model(
                self.config.model_training.dir_out, custom_objects={'_cropped_loss': loss_function})
        else:
            _logger.debug('Building new model, no model exists at {}'.format(
                get_model_filepath(self.config.model_training.dir_out)))
            self.loaded_existing_model = False
            self.model = config_sections.create_model_from_architecture_config_section(
                self.config.model_training.architecture_name, self.config.architecture)
            self.model.compile(loss=loss_function, optimizer=self.config.model_training.optimizer)
        if os.path.exists(get_history_filepath(self.config.model_training.dir_out)):
            assert self.loaded_existing_model, \
                'Model training history exists in model training directory, but existing model found; directory: {}' \
                .format(self.config.model_training.dir_out)
            _logger.debug('Loading existing history from model training directory at {}'.format(
                get_history_filepath(self.config.model_training.dir_out)))
            self.loaded_existing_history = True
            self.history = histories.load_history(get_history_filepath(self.config.model_training.dir_out))
            if 'lr' in self.history:
                learning_rate = self.history['lr'][-1]
                _logger.debug('Setting learning rate to value from last training epoch: {}'.format(learning_rate))
                K.set_value(self.model.optimizer.lr, learning_rate)
        else:
            assert not self.loaded_existing_model, \
                'Trained model exists in model training directory, but no training history found; directory: {}' \
                .format(self.config.model_training.dir_out)
            self.loaded_existing_history = False
            self.history = {'model_name': self.config.model_training.dir_out}
        """    
        # TODO:  reimplement multiple GPUs
        #n_gpu_avail = gpus.get_count_available_gpus()
        #_logger.debug('Using multiple GPUs with {} available'.format(n_gpu_avail))
        n_gpu_avail = 1
        if (n_gpu_avail > 1):
            self._original_model = self.model
            self.model = keras.utils.multi_gpu_model(self._original_model, gpus=n_gpu_avail, cpu_relocation=True)
            self.model.callback_model = self._original_model
            self.model.compile(loss=loss_function, optimizer=self.config.model_training.optimizer)
        """

    def fit_model_with_dataset(self, dataset: Dataset, resume_training: bool = False) -> None:
        return self.fit_model_with_sequences(dataset.training_sequence, dataset.validation_sequence, resume_training)

    def fit_model_with_sequences(
            self,
            training_sequence: BaseSequence,
            validation_sequence: BaseSequence = None,
            resume_training: bool = False
    ):
        if self.history:
            assert resume_training, 'Resume must be true to continue training an existing model'
        if self.config.model_training.assert_gpu:
            gpus.assert_gpu_available()

        model_callbacks = callbacks.get_model_callbacks(self.config, self.history)

        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.
        new_history = self.model.fit_generator(
            training_sequence,
            epochs=self.config.model_training.max_epochs,
            verbose=self.config.model_training.verbosity,
            callbacks=model_callbacks,
            validation_data=validation_sequence,
            max_queue_size=2,
            # workers=psutil.cpu_count(logical=True),
            # use_multiprocessing=True,
            shuffle=False,
            initial_epoch=len(self.history.get('lr', list())),
        )
        self.history = histories.combine_histories(self.history, new_history.history)
        histories.save_history(self.history, get_history_filepath(self.config.model_training.dir_out))
        models.save_model(self.model, get_model_filepath(self.config.model_training.dir_out))

    def calculate_model_memory_footprint(self, batch_size: int) -> float:
        """Calculate model memory footprint. Shamelessly copied from (but not tested rigorously):
        https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model.
        Args:
            batch_size:

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


def get_config_filepath(dir_out: str) -> str:
    """Get the default config path for experiments.

    Args:
        dir_out: Experiment directory.

    Returns:
        Filepath to config.
    """
    return os.path.join(dir_out, configs.DEFAULT_FILENAME_CONFIG)


def get_history_filepath(dir_out: str) -> str:
    """Get the default model training history path for experiments.

    Args:
        dir_out: Experiment directory.

    Returns:
        Filepath to model training history.
    """
    return os.path.join(dir_out, histories.DEFAULT_FILENAME_HISTORY)


def get_model_filepath(dir_out: str) -> str:
    """Get the default model path for experiments.

    Args:
        dir_out: Experiment directory.

    Returns:
        Filepath to model.
    """
    return os.path.join(dir_out, models.DEFAULT_FILENAME_MODEL)
