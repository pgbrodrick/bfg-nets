import os

import keras
import keras.backend as K
import numpy as np

from rsCNN import configs
from rsCNN.data_management.sequences import BaseSequence
from rsCNN.networks import architectures, callbacks, histories, losses, models
from rsCNN.utils import gpus, logging


_logger = logging.get_child_logger(__name__)


# TODO: rename Experiment to something more reasonable, as well as this file
class Experiment(object):
    config = None
    model = None
    history = None
    resume = None
    train_sequence = None
    validation_sequence = None
    test_sequence = None

    def __init__(self, config: configs.Config, resume: bool = False) -> None:
        self.config = config
        self.resume = resume

        if os.path.exists(self.config.model_training.dir_out):
            if histories.load_history(self.config.model_training.dir_out):
                assert self.resume, 'Resume must be true to continue training an existing model'
            # TODO:  do we need to assert that the configs are the same still?
            #original_config = network_configs.load_network_config(self.network_config['model']['dir_out'])
            # differing_items = network_configs.compare_network_configs_get_differing_items(
            #    original_config, self.network_config)
            # assert len(differing_items) == 0, \
            #    'Provided network config differs from network config for existing, trained model:  {}'.format(
            #        differing_items)
        else:
            os.makedirs(self.config.model_training.dir_out)
            configs.save_config_to_file(self.config, self.config.model_training.dir_out)

    def build_or_load_model(self):
        _logger.info('Building or loading model')
        loss_function = losses.cropped_loss(
            self.config.model_training.loss_metric,
            self.config.architecture_options.inshape[0],
            2 * self.config.data_build.loss_window_radius,
            self.config.model_training.weighted
        )
        self.history = histories.load_history(self.config.model_training.dir_out) or dict()
        if self.history:
            _logger.debug('History exists in out directory, loading model from same location')
            self.model = models.load_model(
                self.config.model_training.dir_out, custom_objects={'_cropped_loss': loss_function})
            if 'lr' in self.history:
                _logger.debug('Setting learning rate to value from last training epoch')
                K.set_value(self.model.optimizer.lr, self.history['lr'][-1])
            # TODO:  do we want to warn or raise or nothing if the network type doesn't match the model type?
        else:
            _logger.debug('History does not exist in model out directory, creating new model')
            self.model = architectures.create_model_from_architecture_options(
                self.config.model_training.architecture_name, self.config.architecture_options)
            self.model.compile(loss=loss_function, optimizer=self.config.model_training.optimizer)
            self.history['model_name'] = self.config.model_training.dir_out
        # TODO:  reimplement multiple GPUs
        #n_gpu_avail = gpus.get_count_available_gpus()
        #_logger.debug('Using multiple GPUs with {} available'.format(n_gpu_avail))
        n_gpu_avail = 1
        if (n_gpu_avail > 1):
            self._original_model = self.model
            self.model = keras.utils.multi_gpu_model(self._original_model, gpus=n_gpu_avail, cpu_relocation=True)
            self.model.callback_model = self._original_model
            self.model.compile(loss=loss_function, optimizer=self.config.model_training.optimizer)

    def calculate_training_memory_usage(self, batch_size: int) -> float:
        # Shamelessly copied from
        # https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
        # but not tested rigorously
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

        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes

    def fit_network(
            self,
            train_sequence: BaseSequence,
            validation_sequence: BaseSequence = None
    ) -> None:

        if self.config.model_training.assert_gpu:
            gpus.assert_gpu_available()

        model_callbacks = callbacks.get_callbacks(self.config, self.history)

        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.

        new_history = self.model.fit_generator(
            train_sequence,
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
        histories.save_history(self.history, self.config.model_training.dir_out)
        models.save_model(self.model, self.config.model_training.dir_out)
