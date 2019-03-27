import configparser
import os
from typing import List, Union

import keras
import keras.backend as K
import numpy as np

from rsCNN import utils
from rsCNN.networks import callbacks, history, losses
from rsCNN.data_management import DataConfig, scalers, sequences, training_data, load_data_config_from_file, load_training_data
from rsCNN import evaluation


class Experiment(object):
    data_config = None
    network_config = None
    model = None
    history = None
    train_sequence = None
    validation_sequence = None
    test_sequence = None
    resume = None

    def __init__(self, network_config: configparser.ConfigParser, data_config: DataConfig, resume=False) -> None:
        self.data_config = data_config
        self.network_config = network_config
        self.resume = resume

        if os.path.exists(self.network_config['model']['dir_out']):
            if history.load_history(self.network_config['model']['dir_out']):
                assert self.resume, 'Resume must be true to continue training an existing model'
        else:
            os.makedirs(self.network_config['model']['dir_out'])

        # TODO:  Phil:  this doens't pass the sniff test. The user provides a data_config to __init__ and then we just
        #  overwrite that data_config? 1) Nothing about the __init__ method or the data_config suggests that this
        #  behavior would ever occur, 2) why would the user have a data config that points to a directory where another
        #  data config is located?, 3) this triggers every time someone has resume set to True, which means that it'll
        #  trigger even when a model is running for the first time and a user just wants to be sure it restarts on
        #  preemption. I feel like we should remove this code completely or modify/relocate it if there's a real use
        #  case here.
        """
        # Load up info that already exists from the data config if we're resuming operations
        if (self.resume):
            ldc = load_data_config_from_file(self.data_config.data_save_name)
            if ldc is not None:
                self.data_config = ldc
        """

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

        if (rebuild is False):
            features, responses, weights, read_success = load_training_data(self.data_config)

        if (read_success is False or rebuild is True):
            if (self.data_config.data_build_category == 'ordered_continuous'):
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

        batch_size = self.network_config['training']['batch_size']
        apply_random = self.network_config['training']['apply_random_transformations']
        self.train_sequence = sequences.Sequence([features[_f] for _f in train_folds],
                                                 [responses[_r] for _r in train_folds],
                                                 [weights[_w] for _w in train_folds],
                                                 batch_size,
                                                 self.feature_scaler,
                                                 self.response_scaler,
                                                 apply_random)
        if (self.data_config.validation_fold is not None):
            self.validation_sequence = sequences.Sequence([features[self.data_config.validation_fold]],
                                                          [responses[self.data_config.validation_fold]],
                                                          [weights[self.data_config.validation_fold]],
                                                          batch_size,
                                                          self.feature_scaler,
                                                          self.response_scaler,
                                                          apply_random)
        if (self.data_config.test_fold is not None):
            self.test_sequence = sequences.Sequence([features[self.data_config.test_fold]],
                                                    [responses[self.data_config.test_fold]],
                                                    [weights[self.data_config.test_fold]],
                                                    batch_size,
                                                    self.feature_scaler,
                                                    self.response_scaler,
                                                    apply_random)

    def build_or_load_model(self):
        self.history = history.load_history(self.network_config['model']['dir_out']) or dict()
        loss_function = losses.cropped_loss(
            self.network_config['architecture']['loss_metric'],
            self.data_config.window_radius*2,
            self.data_config.internal_window_radius*2,
            weighted=self.network_config['architecture']['weighted']
        )
        if self.history:
            self.model = history.load_model(
                self.network_config['model']['dir_out'], custom_objects={'_cropped_loss': loss_function})
            # TODO:  do we want to warn or raise or nothing if the network type doesn't match the model type?
        else:
            self.model = self.network_config['architecture']['create_model'](
                self.network_config['architecture']['inshape'],
                self.network_config['architecture']['n_classes'],
                **self.network_config['architecture_options']
            )
            self.model.compile(loss=loss_function, optimizer=self.network_config['training']['optimizer'])

        if self.resume and 'lr' in self.history:
            K.set_value(self.model.optimizer.lr, self.history['lr'][-1])

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

    def evaluate_network(self):
        # TODO: modify to accept sequences
        evaluation.generate_eval_report()

    def fit_network(self) -> None:
        if self.network_config['model']['assert_gpu']:
            utils.assert_gpu_available()

        model_callbacks = callbacks.get_callbacks(self.network_config, self.history)

        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.
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
        self.history = history.combine_histories(self.history, new_history.history)
        history.save_history(self.history, self.network_config['model']['dir_out'])
        history.save_model(self.model, self.network_config['model']['dir_out'])

    def predict(self, features: Union[np.ndarray, List[np.ndarray]]):
        if self.network_config['model']['assert_gpu']:
            utils.assert_gpu_available()

        return self.model.predict(features, batch_size=self.network_config['training']['batch_size'], verbose=0)

    def predict_sequence(self, predict_sequence: keras.utils.Sequence):
        if self.network_config['model']['assert_gpu']:
            utils.assert_gpu_available()

        return self.model.predict_generator(
            predict_sequence,
            max_queue_size=2,
            # workers=psutil.cpu_count(logical=True),
            # use_multiprocessing=True,
            verbose=0
        )
