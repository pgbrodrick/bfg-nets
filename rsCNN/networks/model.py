import configparser
import os
from typing import List, Union

import keras
import keras.backend as K
import numpy as np

from rsCNN import utils
from rsCNN.networks import callbacks, history, losses
from rsCNN.data_management import DataConfig, scalers, sequences


class Experiment(object):
    network_config = None
    _is_model_new = None

    def __init__(self, network_config: configparser.ConfigParser, data_config: DataConfig) -> None:
        """ Initializes the appropriate network

        Arguments:
        network_config -
          Configuration parameter object for the network.
        Keyword Arguments:
        reinitialize - bool
          Flag directing whether the model should be re-initialized from scratch (no weights).
        load_history - bool
          Flag directing whether the model should load it's training history.
        """
        # TODO:  move continue_training here, do all overwrite checks here and reduce all of
        #  the filename/dir settings so that we have one for experiment, only overwrite any objects
        #  if we continue here, change continue_training to new name? below docstring
        """
        assert not (continue_training is False and self._is_model_new is False), \
            'The parameter continue_training must be true to continue training an existing model'

        if continue_training:
            if 'lr' in self.history:
                K.set_value(self.model.optimizer.lr, self.history['lr'][-1])
        """

        self.network_config = network_config
        self.data_config = data_config

        if not os.path.exists(self.network_config['model']['dir_out']):
            os.makedirs(self.network_config['model']['dir_out'])

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
            self._is_model_new = False
        else:
            self.model = self.network_config['architecture']['create_model'](
                self.network_config['architecture']['inshape'],
                self.network_config['architecture']['n_classes'],
                **self.network_config['architecture_options']
            )
            self.model.compile(loss=loss_function, optimizer=self.network_config['training']['optimizer'])
            self._is_model_new = True

    def build_or_load_data(self):
        # TODO:  PHIL:  review the sequence and scalar changes I've made before refactoring this. Things are simple if
        #  you just save the data to disk and keep this independent of other operations. You'll need to integrate
        #  multiple methods and classes if you want to optimize things.
        # TODO:  Phil adds logic for parsing files
        # TODO:  Phil adds logi cfor loading data into features/responses, training munge style
        # TODO:  Phil wants to write logic to find system memory and how much is available to get size
        # TODO:  be sure we return generators/sequences?
        # build data
        # load data
        # create scalers
        # create sequences/generators
        raise
        if True:
            features, responses, weights, fold_assignments = training_data.build_regression_training_data_ordered(
                data_config)
        else:
            data_management = load_training_data(data_config)
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

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
        # TODO:  sequences now have a get_transformed and get_untransformed method (check names) so that you can get
        #  what you need directly. You might need to reorganize that code if youw ant to get both at the same time.
        features, responses = self.test_generator.__getitem__(0)
        evaluation.generate_eval_report(args)

    def fit_network(self) -> None:
        # TODO:  something to think about:  features and responses may not make sense for some architectures, like
        #  those architectures where the inputs and targets of the algorithm are something like inputs = [x]
        #  and targets = [x, y]... that is, where we're trying to reconstruct x while also predicting y. Do we stick
        #  with these names?
        if self.network_config['model']['assert_gpu']:
            utils.assert_gpu_available()

        model_callbacks = callbacks.get_callbacks(self.network_config, self.history)

        # return self.feature_scaler.transform(features), \
        #       self.response_scaler.transform(responses)

        # TODO:  Same parameter questions as with fit()
        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.
        new_history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=1,
            epochs=self.network_config['training']['max_epochs'],
            verbose=self.network_config['model']['verbosity'],
            callbacks=model_callbacks,
            validation_data=self.validation_generator,
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

    def prepare_sequences(self) -> None:
        # TODO:  PHIL:  this takes no time at all, would you prefer these are only in fit or predict methods and that
        #  we don't save references to the sequences? Note that it's useful to save if we want to play around with the
        #  sequences / manually sample, but probably not if we want to keep the object clean of as many attrs or methods
        #  as possible
        batch_size = self.network_config['training']['batch_size']
        apply_random = self.network_config['training']['apply_random_transformations']
        self.train_sequence = sequences.Sequence(batch_size, self.feature_scaler, self.response_scaler, apply_random)
        self.validate_sequence = sequences.Sequence(batch_size, self.feature_scaler, self.response_scaler, apply_random)
        self.test_sequence = sequences.Sequence(batch_size, self.feature_scaler, self.response_scaler)

    def plot_history(self, path_out=None):
        history.plot_history(self.history, path_out)

    def fit_data_scalers(self) -> None:
        # TODO:  incorporate scaler options in data config, might be worth the time to make it similar to how
        #   architectures handle options, since we want to generate templates automatically, but we might need to
        #   have subtemplates for general config, architectures, scalers, and other, since there would be too many
        #   pairwise combinations to have all possibilities pre-generated
        self.feature_scaler = scalers.get_scaler(
            self.data_config.feature_scaler_name, self.data_config.feature_scaler_options)
        self.response_scaler = scalers.get_scaler(
            self.data_config.response_scaler_name, self.data_config.response_scaler_options)
        # TODO:  PHIL:  please figure out how you want to step through the training data files to fit the data scalers,
        #  based on our conversation about needing sufficient numbers of samples to confirm that the transformation is
        #  adequate, or figure out how you want to keep data in memory from building and somehow still fit the
        #  transformers
        self.feature_scaler.fit('TODO')
        self.response_scaler.fit('TODO')
        self.feature_scaler.save()
        self.response_scaler.save()
