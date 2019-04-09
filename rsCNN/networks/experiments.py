import os

import keras
import keras.backend as K
import numpy as np

from rsCNN.data_management import sequences
from rsCNN.networks import callbacks, histories, losses, models, network_configs
from rsCNN.utils import gpus, logger


_logger = logger.get_child_logger(__name__)


class Experiment(object):
    data_config = None
    network_config = None
    model = None
    history = None
    resume = None
    train_sequence = None
    validation_sequence = None
    test_sequence = None

    def __init__(self, network_config: dict, data_config, resume: bool = False) -> None:
        self.data_config = data_config
        self.network_config = network_config
        self.resume = resume

        if os.path.exists(self.network_config['model']['dir_out']):
            if histories.load_history(self.network_config['model']['dir_out']):
                assert self.resume, 'Resume must be true to continue training an existing model'
        else:
            os.makedirs(self.network_config['model']['dir_out'])
            network_configs.save_network_config(network_config, self.network_config['model']['dir_out'])

    def build_or_load_model(self):
        _logger.info('Building or loading model')
        loss_function = losses.cropped_loss(
            self.network_config['architecture']['loss_metric'],
            self.data_config.window_radius*2,
            self.data_config.internal_window_radius*2,
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
            if 'lr' in self.history:
                _logger.debug('Setting learning rate to value from last training epoch')
                K.set_value(self.model.optimizer.lr, self.history['lr'][-1])
        n_gpu_avail = gpus.get_count_available_gpus()
        _logger.debug('Using multiple GPUs with {} available'.format(n_gpu_avail))
        if (n_gpu_avail > 1):
            self._original_model = self.model
            self.model = keras.utils.multi_gpu_model(self._original_model, gpus=n_gpu_avail, cpu_relocation=True)
            if self._original_model is self.model:
                raise AssertionError('The above two lines do not work because the second assignment to self.model ' +
                                     'overwrites self._original_model\'s reference. Tell Nick to change names')
            else:
                raise AssertionError('The above two lines work! Find this line in the codebase and remove the ' +
                                     'surrounding if/else statement. Nick just wanted to be sure it was correct')
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
            train_sequence: keras.utils.Sequence = None,
            validation_sequence: keras.utils.Sequence = None
    ) -> None:
        # TODO:  Phil, I know we're thinking about breaking up the data and network components. I think part of that
        #   might be simplifying this section by just passing in training and validation sequences because there's no
        #   reason for the experiment object to hold onto those, right? If we're just creating a method that sets those
        #   attributes, why not just pass them to the fit or predict method when needed? It's not like experiment will
        #   be doing anything more than just passing to fit and predict, right? In the meantime, I'm leaving the setting
        #   of those attributes here because I don't want to break the combined data/network methods. We'll want to
        #   remove the following lines and then fit directly from the arguments.
        if train_sequence is not None:
            self.train_sequence = train_sequence
        if validation_sequence is not None:
            self.validation_sequence = validation_sequence

        if self.network_config['model']['assert_gpu']:
            gpus.assert_gpu_available()

        model_callbacks = callbacks.get_callbacks(self.network_config, self.history)

        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.

        # new_history = self.parallel_model.fit_generator(
        new_history = self.model.fit_generator(
            self.train_sequence,
            epochs=self.network_config['training']['max_epochs'],
            verbose=self.network_config['model']['verbosity'],
            callbacks=model_callbacks,
            validation_data=self.validation_sequence,
            max_queue_size=2,
            # workers=psutil.cpu_count(logical=True),
            # use_multiprocessing=True,
            shuffle=False,
            initial_epoch=len(self.history.get('lr', list())),
        )
        self.history = histories.combine_histories(self.history, new_history.history)
        histories.save_history(self.history, self.network_config['model']['dir_out'])
        models.save_model(self.model, self.network_config['model']['dir_out'])
