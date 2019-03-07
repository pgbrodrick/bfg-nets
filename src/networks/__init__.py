import os

import keras
import keras.backend as K
import numpy as np


class CNN():

    def __init__(self):
        self.model = None
        self.config = None
        self.training = None

    def create_network(self, network_config):
        """ Initializes the appropriate network

        Arguments:
        net_name - str
          Name of the network to fetch.  Options are:
            flex_unet - a flexible, U-net style network.
        inshape - tuple/list
          Designates the input shape of an image to be passed to
          the network.
        n_classes - int
          The number of classes the network is meant to classify.
        """
        self.config = network_config

        if (self.network_name == 'flex_unet'):
            # Update potentially non-standard network parameters
            self.config.set_default('conv_depth', 16)
            self.config.set_default('batch_norm', False)

            # Return a call to the argument-specific version of flex_unet
            self.model = flex_unet(self.config.inshape,
                                   self.config.n_classes,
                                   self.config['conv_depth'],
                                   self.config['batch_norm']):
        elif (self.network_name == 'flat_regress_net'):

            # Update potentially non-standard network parameters
            self.config.set_default('conv_depth', 16)
            self.config.set_default('batch_norm', False)
            self.config.set_default('n_layers', 8)
            self.config.set_default('conv_pattern', [3])
            self.config.set_default('output_activation', 'softmax')

            self.model = networks.flat_regress_net(NetworkConfig)
            self.model = flex_unet(NetworkConfig.inshape,
                                   self.config.n_classes,
                                   self.config['conv_depth'],
                                   self.config['batch_norm'],
                                   self.config['n_layers'],
                                   self.config['conv_pattern'],
                                   self.config['output_activation']):

        else:
            raise NotImplementedError('Unknown network name')

    def calculate_training_memory_usage(batch_size):
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


# Deprecated.  Let's migrate things upwards as necessary
class NeuralNetwork(object):
    _initial_epoch = 0
    _generator_max_queue_size = 100
    _generator_workers = 1
    _generator_use_multiprocessing = True

    def __init__(self, model=None, dir_load=None, custom_objects=None):
        # If the user created a model, use the model -> unnecessary given the network config now
        if model:
            self.model = model
        # Otherwise load up the model and history from a previously trained model
        elif dir_load:
            self.model = keras.models.load_model(
                os.path.join(dir_load, self._filename_model), custom_objects=custom_objects)
            with open(os.path.join(dir_load, self._filename_history), 'rb') as file_:
                callbacks.load_history(file_)
            self._initial_epoch = len(self.history['lr'])
            K.set_value(self.model.optimizer.lr, self.history['lr'][-1])

    def fit_sequence(self, train_sequence, dir_out, max_training_epochs, validation_sequence=None):
        # Prep callbacks with dynamic parameters
        # Train model
        self.model.fit_generator(
            train_sequence, epochs=max_training_epochs, callbacks=callbacks, validation_data=validation_sequence,
            max_queue_size=self._generator_max_queue_size, use_multiprocessing=self._generator_use_multiprocessing,
            workers=self._generator_workers, initial_epoch=self._initial_epoch, verbose=self._verbosity,
        )

    def evaluate(self, evaluate_sequence):
        return self.model.evaluate_generator(
            evaluate_sequence, max_queue_size=self._generator_max_queue_size,
            use_multiprocessing=self._generator_use_multiprocessing, workers=self._generator_workers,
            verbose=self._verbosity
        )

    def predict(self, inputs):
        return self.model.predict(inputs, batch_size=None, verbose=self._verbosity)

    def predict_sequence(self, predict_sequence):
        return self.model.predict_generator(
            predict_sequence, max_queue_size=self._generator_max_queue_size,
            use_multiprocessing=self._generator_use_multiprocessing, workers=self._generator_workers,
            verbose=self._verbosity
        )
