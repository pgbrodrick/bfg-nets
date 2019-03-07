import datetime
import json
import os

import keras
import keras.backend as K
import numpy as np


class NetworkConfig:
    """ A wrapper class designed to hold all relevant configuration information for the
        training of a new network.  
    """

    def __init__(self, network_type, inshape, outshape, dir_out, **kwargs):
        """ 
          Arguments:
          network_type - str
            Style of the network to use.  Options are:
              flex_unet
              flat_regress_net
          inshape - tuple/list
            Designates the input shape of an image to be passed to
            the network.
          outshape - tuple/list
            Designates the output shape of targets to be fit by the network
          dir_out - str
            Directory for network output
        """
        self.network_type = network_type
        self.inshape = inshape
        self.outshape = outshape
        self.dir_out = dir_out

        # Optional arguments
        self.filepath_model_out = kwargs.get('filepath_model_out', './model.h5')
        self.filepath_history_out = kwargs.get('filepath_history_out', './history.json')
        self.checkpoint_periods = kwargs.get('checkpoint_periods', 5)
        self.verbosity = kwargs.get('verbosity', 1)
        self.append_existing = kwargs.get('append_existing', False)

        # Callbacks
        self.callbacks_use_tensorboard = kwargs.get('callbacks_use_tensorboard', True)
        self.filepath_tensorboard_out = kwargs.get('dir_tensorboard_out', './tensorboard')
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


def get_callbacks(network_config):
    if network_config.append_existing:
        existing_history = load_history(network_config.filepath_history_out)
    else:
        existing_history = dict()
    callbacks = [
        HistoryCheckpoint(
            network_config.filepath_history_out,
            existing_history=existing_history,
            period=network_config.checkpoint_periods,
            verbose=network_config.verbosity
        ),
        keras.callbacks.ModelCheckpoint(
            network_config.filepath_model_out,
            period=network_config.checkpoint_periods,
            verbose=network_config.verbosity
        ),
    ]
    if network_config.callbacks_use_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=network_config.early_stopping_min_delta,
                patience=network_config.early_stopping_patience
            ),
        )
    if network_config.callbacks_use_reduced_learning_rate:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=network_config.reduced_learning_rate_factor,
                min_delta=network_config.reduced_learning_rate_min_delta,
                patience=network_config.reduced_learning_rate_patience
            ),
        )
    if network_config.callbacks_use_tensorboard:
        callbacks.append(
            keras.callbacks.TensorBoard(
                network_config.filepath_tensorboard_out,
                histogram_freq=network_config.tensorboard_histogram_freq,
                write_graph=network_config.tensorboard_write_graph,
                write_grads=network_config.tensorboard_write_grads,
                write_images=network_config.tensorboard_write_images,
                update_freq=network_config.tensorboard_update_freq,
            ),
        )
    if network_config.callbacks.use_terminate_on_nan:
        callbacks.append(keras.callbacks.TerminateOnNaN())
    return callbacks


def load_history(filepath):
    with open(filepath, 'r') as file_:
        history = json.load(file_)
    return history



# TODO - Fabina, can you populate this with the useful info you want to retain from training?
class training_history:
    """ A wrapper class designed to hold all relevant configuration information obtained
        during training/testing the model.  
    """


class CNN():

    def __init__(self):
        self.model = None
        self.config = None
        self.training = None

    def create_config(self, network_name, input_shape, n_classes, network_file=None, network_dictionary=None):
        self.config = NetworkConfig(network_name, input_shape, n_classes)

        if (network_file is not None):
            self.config.read_from_file(network_file)

        if (network_dictionary is not None):
            self.config.read_from_dict(network_dictionary)

    def create_network(self):
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
    _verbosity = 1
    _initial_epoch = 0
    _generator_max_queue_size = 100
    _generator_workers = 1
    _generator_use_multiprocessing = True

    def __init__(self, model=None, dir_load=None, custom_objects=None):
        assert model is None or dir_load is None, 'Must either pass compiled model or directory with saved model data'
        if model:
            self.model = model
        elif dir_load:
            self.model = keras.models.load_model(
                os.path.join(dir_load, self._filename_model), custom_objects=custom_objects)
            with open(os.path.join(dir_load, self._filename_history), 'rb') as file_:
                self.history = pickle.load(file_)
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


class HistoryCheckpoint(keras.callbacks.Callback):

    def __init__(self, filepath, existing_history=None, period=1, verbose=0):
        super().__init__()
        self.filepath = filepath
        if existing_history is None:
            existing_history = dict()
        self.existing_history = existing_history
        self.period = period
        self.verbose = verbose
        self.epochs_since_last_save = 0
        self.epoch_begin = None

    def on_train_begin(self, logs=None):
        for key in ('epoch_start', 'epoch_finish'):
            self.existing_history.setdefault(key, list())
        super().on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        # Update times
        epoch_end = datetime.datetime.now()
        self.existing_history['epoch_start'].append(self.epoch_begin)
        self.existing_history['epoch_finish'].append(epoch_end)
        self.epoch_begin = None
        # Save if necessary
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        history = self._format_history(self.model.history, self.existing_history)
        with open(self.filepath, 'wb') as file_:
            pickle.dump(history, file_)
        if self.verbose > 0:
            print('\nEpoch %05d: saving history to %s' % (epoch + 1, self.filepath))

    def _format_history(self, new_history, old_history=None):
        if old_history is None:
            old_history = dict()
        combined_history = old_history.copy()
        for key, value in new_history.history.items():
            combined_history.setdefault(key, list()).extend(value)
        return combined_history
