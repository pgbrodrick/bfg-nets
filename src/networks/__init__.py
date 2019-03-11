import os

import keras
import keras.backend as K
import numpy as np

from src.networks import architectures, callbacks, history


class TrainingHistory(object):
    """ A wrapper class designed to hold all relevant configuration information obtained
        during training/testing the model.
    """
    # TODO - Fabina, can you populate this with the useful info you want to retain from training?
    # TODO - Phil:  let's punt until we know what we need and what everything else looks like?
    pass


class NetworkConfig(object):
    """ A wrapper class designed to hold all relevant configuration information for the
        training of a new network.
    """

    def __init__(self, network_type, inshape, n_classes, **kwargs):
        """
          Arguments:
          network_type - str
            Style of the network to use.  Options are:
              flex_unet
              flat_regress_net
          inshape - tuple/list
            Designates the input shape of an image to be passed to
            the network.
          n_classes - tuple/list
            Designates the output shape of targets to be fit by the network
        """
        self.network_type = network_type
        self.inshape = inshape
        self.n_classes = n_classes

        if (self.network_type == 'flex_unet'):
            self.create_architecture = architectures.unet.flex_unet
            self.architecture_options = {
                'conv_depth': kwargs.get('conv_depth', 16),
                'batch_norm': kwargs.get('batch_norm', False),
            }
        elif (self.network_type == 'flat_regress_net'):
            self.create_architecture = architectures.regress_net.flat_regress_net
            self.architecture_options = {
                'conv_depth': kwargs.get('conv_depth', 16),
                'batch_norm': kwargs.get('batch_norm', False),
                'n_layers': kwargs.get('n_layers', 8),
                'conv_pattern': kwargs.get('conv_pattern', [3]),
                'output_activation': kwargs.get('output_activation', 'softmax'),
            }
        else:
            NotImplementedError('Invalid network type: ' + self.network_type)

        # Optional arguments
        self.dir_out = kwargs.get('dir_out', './')
        self.filepath_model_out = kwargs.get('filepath_model_out', 'model.h5')
        self.filepath_history_out = kwargs.get('filepath_history_out', 'history.json')
        self.checkpoint_periods = kwargs.get('checkpoint_periods', 5)
        self.verbosity = kwargs.get('verbosity', 1)
        self.append_existing = kwargs.get('append_existing', False)

        # Training arguments
        self.batch_size = 1
        self.max_epochs = 100
        self.n_noimprovement_repeats = 10
        self.output_directory = None  # TODO: give a default
        self.verification_fold = None

        # Callbacks
        self.callbacks_use_tensorboard = kwargs.get('callbacks_use_tensorboard', True)
        self.filepath_tensorboard_out = kwargs.get('dir_tensorboard_out', 'tensorboard')
        self.tensorboard_update_freq = kwargs.get('tensorboard_update_freq', 'epoch')
        self.tensorboard_histogram_freq = kwargs.get('tensorboard_histogram_freq', 0)
        self.tensorboard_write_graph = kwargs.get('tensorboard', True)
        self.tensorboard_write_grads = kwargs.get('tensorboard', False)
        self.tensorboard_write_images = kwargs.get('tensorboard', True)

        self.callbacks_use_early_stopping = kwargs.get('callbacks_use_early_stopping', True)
        self.early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 10**-4)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 50)

        self.callbacks_use_reduced_learning_rate = kwargs.get('callbacks_use_reduced_learning_rate', True)
        self.reduced_learning_rate_factor = kwargs.get('reduced_learning_rate_factor', 0.5)
        self.reduced_learning_rate_min_delta = kwargs.get('reduced_learning_rate_min_delta', 10**-4)
        self.reduced_learning_rate_patience = kwargs.get('reduced_learning_rate_patience', 10)

        self.callbacks_use_terminate_on_nan = kwargs.get('terminate_on_nan', True)


class CNN():

    def __init__(self, network_config):
        """ Initializes the appropriate network

        Arguments:
        net_name - str
          Name of the network to fetch.  Options are:
            flex_unet - a flexible, U-net style network.
        inshape - tuple/list
          Designates the input shape of an image to be passed to
          the network.
        n_classes - int
          Designates the number of response layers.
        """
        self.config = network_config
        self.model = None
        self.training = None
        self.model = self.config.create_architecture(**self.config.architecture_options)

    def calculate_training_memory_usage(self, batch_size):
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

    # TODO during fit, make sure that all training_options (as well as network options) are saved with the model
    def fit(self, features, responses, fold_assignments, verbose=True):
        model_callbacks = callbacks.get_callbacks(self.config)

        if (self.config.verification_fold is not None):
            train_subset = fold_assignments == self.config.verification_fold
            test_subset = np.logical_not(train_subset)
            train_features = features[train_subset, ...]
            train_responses = responses[train_subset]
            validation_data = (features[test_subset, ...], responses[test_subset, ...])
        else:
            train_features = features
            train_responses = responses
            validation_data = None

        self.model.fit(train_features,
                       train_responses,
                       validation_data=validation_data,
                       epochs=self.config.max_epochs,
                       batch_size=self.config.batch_size,
                       verbose=verbose,
                       shuffle=False,
                       callbacks=model_callbacks)

    def fit_sequence(self, train_sequence, dir_out, max_training_epochs, validation_sequence=None):
        # Prep callbacks with dynamic parameters
        # Train model
        self.model.fit_generator(
            train_sequence, epochs=max_training_epochs, callbacks=callbacks, validation_data=validation_sequence,
            max_queue_size=self._generator_max_queue_size, use_multiprocessing=self._generator_use_multiprocessing,
            workers=self._generator_workers, initial_epoch=self._initial_epoch, verbose=self._verbosity,
        )


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
