import configparser
import os
from typing import List, Union

import keras
import keras.backend as K
import numpy as np

from rsCNN import utils
from rsCNN.data_management import scalers, sequences, training_data, load_data_config_from_file, load_training_data
from rsCNN.networks import callbacks, io, losses
from rsCNN.utils import logger


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

    def __init__(self, network_config: configparser.ConfigParser, data_config, resume=False) -> None:
        self.data_config = data_config
        self.network_config = network_config
        self.resume = resume

        if os.path.exists(self.network_config['model']['dir_out']):
            if io.load_history(self.network_config['model']['dir_out']):
                assert self.resume, 'Resume must be true to continue training an existing model'
        else:
            os.makedirs(self.network_config['model']['dir_out'])

    def build_or_load_data(self, rebuild=False):
        """
            This function does the following, considering the rebuild parameter at each step:
                1) load/build training data
                2) load/initialize/fit scalers
                3) initiate train/validation/test sequences as components of Experiment
        """
        # TODO:  Phil:  thinking about this more, data_config is only used in two places, this method and the
        #  architecture
        # TODO: start off by checking to make sure that we have all requisite info via assert
        # Load up info that already exists from the data config if we're resuming operations
        if (self.resume):
            ldc = load_data_config_from_file(self.data_config.data_save_name)
            if ldc is not None:
                self.data_config = ldc

        if (rebuild is False):
            features, responses, weights, read_success = load_training_data(self.data_config)

        if (read_success is False or rebuild is True):
            if (self.data_config.data_build_category in ['ordered_continuous','ordered_categorical'] ):
                features, responses, weights = training_data.build_training_data_ordered(self.data_config)
            else:
                raise NotImplementedError('Unknown data_build_category')

        # TODO:  incorporate scaler options in data config, might be worth the time to make it similar to how
        #   architectures handle options, since we want to generate templates automatically, but we might need to
        #   have subtemplates for general config, architectures, scalers, and other, since there would be too many
        #   pairwise combinations to have all possibilities pre-generated
        feat_scaler_atr = {'nodata_value': self.data_config.feature_nodata_value,
                           'savename_base': self.data_config.data_save_name + '_feature_scaler'}
        self.feature_scaler = scalers.get_scaler(self.data_config.feature_scaler_name,
                                                 feat_scaler_atr)

        resp_scaler_atr = {'nodata_value': self.data_config.response_nodata_value,
                           'savename_base': self.data_config.data_save_name + '_response_scaler'}
        self.response_scaler = scalers.get_scaler(self.data_config.response_scaler_name,
                                                  resp_scaler_atr)
        self.feature_scaler.load()
        self.response_scaler.load()

        train_folds = [x for x in np.arange(
            self.data_config.n_folds) if x is not self.data_config.validation_fold and x is not self.data_config.test_fold]

        if (self.feature_scaler.is_fitted is False or rebuild is True):
            # TODO: do better
            self.feature_scaler.fit(features[train_folds[0]])
            self.feature_scaler.save()
        if (self.response_scaler.is_fitted is False or rebuild is True):
            # TODO: do better
            self.response_scaler.fit(responses[train_folds[0]])
            self.response_scaler.save()

        # TODO: Fabina, can't see where you're loading this in from now, presumably,
        # it's in a config somewhere?
        #batch_size = self.network_config['training']['batch_size']
        #batch_size = self.network_config['training']['batch_size']
        batch_size = 10
        apply_random = self.network_config['training']['apply_random_transformations']
        mean_centering = self.data_config.feature_mean_centering
        self.train_sequence = sequences.MemmappedSequence([features[_f] for _f in train_folds],
                                                          [responses[_r] for _r in train_folds],
                                                          [weights[_w] for _w in train_folds],
                                                          self.feature_scaler,
                                                          self.response_scaler,
                                                          batch_size,
                                                          apply_random_transforms = apply_random,
                                                          feature_mean_centering = mean_centering)
        if (self.data_config.validation_fold is not None):
            self.validation_sequence = sequences.MemmappedSequence([features[self.data_config.validation_fold]],
                                                                   [responses[self.data_config.validation_fold]],
                                                                   [weights[self.data_config.validation_fold]],
                                                                   self.feature_scaler,
                                                                   self.response_scaler,
                                                                   batch_size,
                                                                   apply_random_transforms = apply_random,
                                                                   feature_mean_centering = mean_centering)
        if (self.data_config.test_fold is not None):
            self.test_sequence = sequences.MemmappedSequence([features[self.data_config.test_fold]],
                                                             [responses[self.data_config.test_fold]],
                                                             [weights[self.data_config.test_fold]],
                                                             self.feature_scaler,
                                                             self.response_scaler,
                                                             batch_size,
                                                             apply_random_transforms = apply_random,
                                                             feature_mean_centering = mean_centering)

    def build_or_load_model(self):
        _logger.info('Building or loading model')
        loss_function = losses.cropped_loss(
            self.network_config['architecture']['loss_metric'],
            self.data_config.window_radius*2,
            self.data_config.internal_window_radius*2,
            weighted=self.network_config['architecture']['weighted']
        )
        self.history = io.load_history(self.network_config['model']['dir_out']) or dict()
        if self.history:
            _logger.debug('History exists in out directory, loading model from same location')
            self.model = io.load_model(
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
        n_gpu_avail = utils.num_gpu_available()
        print('There are ' + str(n_gpu_avail) + ' gpus available')
        #if (n_gpu_avail > 1):
        #    self.parallel_model = keras.utils.multi_gpu_model(self.model, gpus=n_gpu_avail, cpu_relocation=True)
        #    self.parallel_model.compile(loss=loss_function, optimizer=self.network_config['training']['optimizer'])
        #else:
        #    self.parallel_model = self.model


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
            utils.assert_gpu_available()

        model_callbacks = callbacks.get_callbacks(self.network_config, self.history)

        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.

        #new_history = self.parallel_model.fit_generator(
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
        self.history = io.combine_histories(self.history, new_history.history)
        io.save_history(self.history, self.network_config['model']['dir_out'])
        io.save_model(self.model, self.network_config['model']['dir_out'])
