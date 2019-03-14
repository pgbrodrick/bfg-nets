import keras.backend as K
import numpy as np
import os
import warnings

import keras

from rsCNN.networks import callbacks, history, losses, network_config
from rsCNN.utils import assert_gpu_available


class TrainingHistory(object):
    """ A wrapper class designed to hold all relevant configuration information obtained
        during training/testing the model.
    """
    # TODO - Fabina, can you populate this with the useful info you want to retain from training?
    # TODO - Phil:  let's punt until we know what we need and what everything else looks like?
    pass


class CNN(object):

    def __init__(
        self,
        network_config: network_config.NetworkConfig,
        load_history: bool = True,
        reinitialize: bool = False
    ) -> None:
        """ Initializes the appropriate network

        Arguments:
        network_config - NetworkConfig
          Configuration parameter object for the network.
        TODO:  I'm looking at this and thinking about the use cases from my Github comment and that makes me think
         about this class more generally. What's the purpose of this class? Well, it's meant to manage the package
         of items that are associated with a keras model in a proper experimental workflow. It's not meant to be a
         filesystem manager (for cleaning up models), right?
         So what are the use cases for this object?
           1. Start a new model where no model exists.
           2. Start a new model despite a model already existing.
           3. Continue training a model that exited without finishing. This is requires a load and a continuation from
              the last training state.
           4. Continue training a model that finished, e.g., transfer learning. This requires a load and a fresh train
              state.
           5. Load an existing model to use for prediction or to otherwise introspect. This requires a load.
         Are there others?
         Organizational observations:
           - Cases 2-5 require a model exists at the filepath, thus case 1 can be safely assumed if no model exists.
           - Given that no model exists, case 1 needs no special arguments or checks because it is non-destructive
           - Case 2 requires special arguments because it is destructive. It overwrites a model in the defined
              location. However, I would argue that perhaps we shouldn't introduce a overwrite parameter for the
              following reason. The default in our config files is likely to have overwrite disabled because it's
              safer, right? So to overwrite, one needs to open the config file and modify the overwrite option, and
              then it's likely that one would need to go in afterwards to modify that overwrite option to protect the
              newly generated model. This multi-step workflow replaces the single step of removing the existing
              model via the command line and is more likely to result in unintended consequences (i.e., forgetting to
              reset the overwrite option in the config file). Moreover, we already have a ton of configuration
              options, we're going to be introducing more, and we probably only want to introduce them where
              necessary. Given this, I would suggest we do not support Case 2 explicitly.
           - Case 3 and case 4 differ in two respects. In case 3, the previously trained model is incomplete and
              overwrites are not a concern, and that histories should be appended. What I mean by appending history is
              that, for instance, when we're saving learning rates, we want to append the new learning rates to the
              old because they're directly comparable and the sequence is important. In case 4, the previously trained
              model is complete and may be valuable on its own, so overwrites are undesirable. Also, for details like
              learning rates, we want to know there has been a "reset" or "fresh start" for plots and review. We could
              add in several additional flags for this, but we could also handle it automatically with two changes:
              1) saving models in a new location (e.g., modifying model name with datetime) so that saving models is
              not destructive and 2) adding a flag for "resetting" the training phase by resetting learning rates,
              which is then used to determine whether the history object appends information to the previous history
              data or creates a new list for data within the same history object.
           - Case 5 is a safe operation if the user uses predict() as required. It fits with the strategy of
              supporting cases 1, 3, and 4, i.e., loading models when they exist and initializing a new model when
              they do not exist, and it fails clearly and quickly when you have not loaded a model and attempt to
              predict. The user cannot destroy state by predict(), but can destroy state by accidentally calling
              fit(). However, this case is covered by the strategy above for handling cases 3 and 4, where previous
              models are default maintained and new models are created when continuing learning.
         Proposal:
          - Always load the most recent model (by datetime in the filename) if one exists and have no arguments for
          loading
          - Always save new models using the start datetime of the current run, never overwrite a previous model and
          require the user does this themselves
          - Only use a flag for whether training should reset learning rates for "new" training or continue from
          where it left off... but only in the fit() method and not initialization. We default to continuing because
          this finishes immediately if the model was trained already and allows the user to quickly fix if this was
          unintended. If the default was to reset, the model would train fully and the user may not notice for hours if
          this was unintended.
        """
        self.config = network_config

        if (load_history and not reinitialize):
            warnings.warn('Warning: loading model history and re-initializing the model')

        if (reinitialize == False and os.path.isfile(self.config.filepath_model)):
            self.model = keras.models.load_model(self.config.filepath_model)
        else:
            self.model = self.config.create_model(
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
