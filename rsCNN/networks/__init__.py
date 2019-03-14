import keras.backend as K
import numpy as np
import os

from rsCNN.networks import callbacks, history, network_config
from rsCNN.utils import assert_gpu_available


class TrainingHistory(object):
    """ A wrapper class designed to hold all relevant configuration information obtained
        during training/testing the model.
    """
    # TODO - Fabina, can you populate this with the useful info you want to retain from training?
    # TODO - Phil:  let's punt until we know what we need and what everything else looks like?
    pass


class CNN(object):

    def __init__(self, network_config: network_config.NetworkConfig, reinitialize=False):
        """ Initializes the appropriate network

        Arguments:
        network_config - NetworkConfig
          Configuration parameter object for the network.

        Keyword Arguments:
        reinitialize - bool
          Flag directing whether the model should be re-initialized from scratch (no weights).
        load_history - bool
          Flag directing whether the model should load it's training history.
        """
        self.config = network_config

        if (load_history and not reinitialize):
            warning.warn('Warning: loading model history and re-initializing the model')

        if (reinitialize == False and os.path.isfile(self.config.filepath_model_out)):
            self.model = keras.models.load_model(self.config.filepath_model)
        else:
            self.model = self.config.create_architecture(
                self.config.inshape, self.config.n_classes, **self.config.architecture_options)

        self.model.compile(loss=self.config.loss_function, optimizer=self.config.optimizer)

        self.history = dict()
        self.training = None
        self.initial_epoch = 0

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

    def fit(self, features, responses, fold_assignments, load_history=True):
        if self.config.assert_gpu:
            assert_gpu_available()

        if (load_history and os.path.isfile(self.config.filepath_history)):
            history.load_history(self.config.filepath_history)
            self.initial_epoch = len(self.history['lr'])

            # TODO: check into if this is legit, I think it probably is the right call
            K.set_value(self.model.optimizer.lr, self.history['lr'][-1])

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
                       verbose=self.config.verbosity,
                       shuffle=False,
                       initial_epoch=self.intial_epoch,
                       callbacks=model_callbacks)

    def fit_sequence(self, train_sequence, validation_sequence=None):
        # TODO:  reimplement if/when we need generators, ignore for now
        raise NotImplementedError

    def predict(self, features):
        # TODO: Fabina, the verbosity below could be a config parameter, but you basically always want this off (it's either super
        # fast or we're running some structured read/write that has an external reporting setup)
        return self.model.predict(features, batch_size=self.config.batch_size, verbose=False)

    def predict_sequence(self, predict_sequence):
        # TODO:  reimplement if/when we need generators, ignore for now
        raise NotImplementedError
