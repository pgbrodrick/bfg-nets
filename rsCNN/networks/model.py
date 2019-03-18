import configparser
import keras
import keras.backend as K
import numpy as np
import os
from typing import List, Union

from rsCNN import utils
from rsCNN.networks import callbacks, history, losses


class TrainingHistory(object):
    """ A wrapper class designed to hold all relevant configuration information obtained
        during training/testing the model.
    """
    # TODO - Fabina, can you populate this with the useful info you want to retain from training?
    # TODO - Phil:  let's punt until we know what we need and what everything else looks like?
    pass


class CNN(object):
    network_config = None
    _is_model_new = None

    def __init__(self, network_config: configparser.ConfigParser) -> None:
        """ Initializes the appropriate network

        Arguments:
        network_config -
          Configuration parameter object for the network.
        """
        self.network_config = network_config

        if not os.path.exists(self.network_config['model']['dir_out']):
            os.makedirs(self.network_config['model']['dir_out'])

        self.history = history.load_history(self.network_config['model']['dir_out']) or dict()
        # TODO:  this needs a reference to a loss function str, but it also needs the window information and others
        #  I'm just getting around this now by using the inshape and dividing it in half, and we'll address it later
        #  when Phil has the config fully fleshed out.
        loss_function = losses.cropped_loss(
            self.network_config['architecture']['loss_function'],
            self.network_config['architecture']['inshape'][0],
            int(self.network_config['architecture']['inshape'][0] / 2),
            weighted=False
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

    def fit(
            self,
            train_features: Union[np.ndarray, List[np.ndarray]],
            train_responses: Union[np.ndarray, List[np.ndarray]],
            validation_split: float = None,
            validation_data: tuple = None,
            continue_training: bool = False
    ) -> None:
        if self.network_config['model']['assert_gpu']:
            utils.assert_gpu_available()

        assert not (continue_training is False and self._is_model_new is False), \
            'The parameter continue_training must be true to continue training an existing model'

        if continue_training:
            if 'lr' in self.history:
                K.set_value(self.model.optimizer.lr, self.history['lr'][-1])

        model_callbacks = callbacks.get_callbacks(self.network_config, self.history)
        # TODO:  Do we need the flexibility to set steps_per_epoch, validation_steps, or validation_freq, or are there
        #  obvious and reasonable defaults to just use? w.r.t. validation steps and freq, I can only think of
        #  reasons to change them based on computational resource budgets.
        new_history = self.model.fit(
            train_features,
            train_responses,
            batch_size=self.network_config['training']['batch_size'],
            epochs=self.network_config['training']['max_epochs'],
            verbose=self.network_config['model']['verbosity'],
            callbacks=model_callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=False,
            initial_epoch=len(self.history.get('lr', list())),
            steps_per_epoch=1,
            validation_steps=1,
            validation_freq=1
        )
        self.history = history.combine_histories(self.history, new_history)
        history.save_history(self.history, self.network_config['model']['dir_out'])
        history.save_model(self.model, self.network_config['model']['dir_out'])

    def fit_generator(
            self,
            train_generator: keras.utils.Sequence,
            validation_generator: keras.utils.Sequence = None,
            continue_training: bool = False
    ) -> None:
        if self.network_config['model']['assert_gpu']:
            utils.assert_gpu_available()

        assert not (continue_training is False and self._is_model_new is False), \
            'The parameter continue_training must be true to continue training an existing model'

        if continue_training:
            if 'lr' in self.history:
                K.set_value(self.model.optimizer.lr, self.history['lr'][-1])

        model_callbacks = callbacks.get_callbacks(self.network_config, self.history)

        # TODO:  Same parameter questions as with fit()
        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.
        new_history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=1,
            epochs=self.network_config['training']['max_epochs'],
            verbose=self.network_config['model']['verbosity'],
            callbacks=model_callbacks,
            validation_data=validation_generator,
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

    def plot_history(self, path_out=None):
        history.plot_history(self.history, path_out)
