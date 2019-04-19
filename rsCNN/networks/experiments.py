import os

import keras
import keras.backend as K
import numpy as np

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.networks import callbacks, histories, losses, models, network_configs
from rsCNN.utils import gpus, logging


_logger = logging.get_child_logger(__name__)


# TODO: rename Experiment to something more reasonable, as well as this file
class Experiment(object):
    data_config = None
    network_config = None
    model = None
    history = None
    resume = None
    train_sequence = None
    validation_sequence = None
    test_sequence = None

    def __init__(self, network_config: dict, resume: bool = False) -> None:
        self.network_config = network_config
        self.resume = resume

        if os.path.exists(self.network_config['model']['dir_out']):
            if histories.load_history(self.network_config['model']['dir_out']):
                assert self.resume, 'Resume must be true to continue training an existing model'
            #original_config = network_configs.load_network_config(self.network_config['model']['dir_out'])
            # differing_items = network_configs.compare_network_configs_get_differing_items(
            #    original_config, self.network_config)
            # assert len(differing_items) == 0, \
            #    'Provided network config differs from network config for existing, trained model:  {}'.format(
            #        differing_items)
        else:
            os.makedirs(self.network_config['model']['dir_out'])
            network_configs.save_network_config(network_config, self.network_config['model']['dir_out'])

    def build_or_load_model(self):
        _logger.info('Building or loading model')
        loss_function = losses.cropped_loss(
            self.network_config['architecture']['loss_metric'],
            self.network_config['architecture']['inshape'][0],
            self.network_config['architecture']['internal_window_radius']*2,
            weighted=self.network_config['architecture']['weighted']
        )
        self.history = histories.load_history(self.network_config['model']['dir_out']) or dict()
        if self.history:
            _logger.debug('History exists in out directory, loading model from same location')
            self.model = models.load_model(
                self.network_config['model']['dir_out'], custom_objects={'_cropped_loss': loss_function})
            # TODO:  do we want to warn or raise or nothing if the network type doesn't match the model type?
        else:
            _logger.debug('History does not exists in out directory, creating new model')
            self.model = self.network_config['architecture']['create_model'](
                self.network_config['architecture']['inshape'],
                self.network_config['architecture']['n_classes'],
                **self.network_config['architecture_options']
            )
            self.model.compile(loss=loss_function, optimizer=self.network_config['training']['optimizer'])
            self.history['model_name'] = self.network_config['model']['dir_out']
            # TODO:  PHIL, I THINK THIS IS WRONG, BUT I'M NOT CONFIDENT IN MY FRIDAY NIGHT CODING, SHOULD THIS BE SET
            #  IN THE OTHER BRANCH OF THE IF/ELSE?
            if 'lr' in self.history:
                _logger.debug('Setting learning rate to value from last training epoch')
                K.set_value(self.model.optimizer.lr, self.history['lr'][-1])
        n_gpu_avail = gpus.get_count_available_gpus()
        n_gpu_avail = 1
        _logger.debug('Using multiple GPUs with {} available'.format(n_gpu_avail))
        if (n_gpu_avail > 1):
            self._original_model = self.model
            self.model = keras.utils.multi_gpu_model(self._original_model, gpus=n_gpu_avail, cpu_relocation=True)
            self.model.callback_model = self._original_model
            self.model.compile(loss=loss_function, optimizer=self.network_config['training']['optimizer'])

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

        if self.network_config['model']['assert_gpu']:
            gpus.assert_gpu_available()

        model_callbacks = callbacks.get_callbacks(self.network_config, self.history)

        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.

        new_history = self.model.fit_generator(
            train_sequence,
            epochs=self.network_config['training']['max_epochs'],
            verbose=self.network_config['model']['verbosity'],
            callbacks=model_callbacks,
            validation_data=validation_sequence,
            max_queue_size=2,
            # workers=psutil.cpu_count(logical=True),
            # use_multiprocessing=True,
            shuffle=False,
            initial_epoch=len(self.history.get('lr', list())),
        )
        self.history = histories.combine_histories(self.history, new_history.history)
        histories.save_history(self.history, self.network_config['model']['dir_out'])
        models.save_model(self.model, self.network_config['model']['dir_out'])
