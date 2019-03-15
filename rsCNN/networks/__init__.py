import keras
import keras.backend as K
import numpy as np
import os
import psutil
from typing import List, Union

from rsCNN import utils
from rsCNN.networks import callbacks, history, losses, network_config


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

    def __init__(self, network_config: network_config.NetworkConfig,) -> None:
        """ Initializes the appropriate network

        Arguments:
        network_config - NetworkConfig
          Configuration parameter object for the network.
        """
        self.network_config = network_config

        path_base = os.path.join(self.network_config.dir_out, self.network_config.model_name)
        if not os.path.exists(path_base):
            os.makedirs(path_base)

        self.history = history.load_history(path_base) or dict()
        if self.history:
            # TODO:  we need to automatically know what custom_objects are OR save custom_objects along with the model,
            #  these will probably? just be loss functions; this will fail until populated
            self.model = history.load_model(path_base, self.network_config.custom_objects)
            # TODO:  do we want to warn or raise or nothing if the network type doesn't match the model type?
            self._is_model_new = False
        else:
            self.model = self.network_config.create_model(
                self.network_config.inshape, self.network_config.n_classes, **self.network_config.architecture_options)
            self.model.compile(loss=self.network_config.loss_function, optimizer=self.network_config.optimizer)
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
            train_features: Union[np.ndarray, List[np.ndarray, ...]],
            train_responses: Union[np.ndarray, List[np.ndarray, ...]],
            validation_split: float = None,
            validation_data: tuple = None,
            continue_training: bool = False
    ) -> None:

        self._run_checks_before_model_fit(continue_training)
        if continue_training:
            self._set_model_initial_learning_rate_from_last_epoch()
        model_callbacks = callbacks.get_callbacks(self.network_config)

        # TODO:  Does it make sense to have verification fold defined in the config and fold assignments passed here?
        #  this seems a little unusual, where you need to know your verification fold beforehand but you can change
        #  your fold assignments on the fly. I'm wondering whether it should even be the responsibility of the fit
        #  function to *also* handle the data folds. Perhaps this should be handled at a higher level? I'm thinking
        #  of the case where you want to iterate through all of your folds. It seems a little cleaner to just pass
        #  the correct training and validation values to this function, so that you can change your verification fold
        #  on the fly without changing your fold assignments. I'm guessing you're probably on board because
        #  it looks like we have a few logic errors here, so we must have been just writing this quickly (e.g., it looks
        #  like fold_assignments is required even though it may not be used). I've removed the code from this section
        #  and we can either 1) add that code here again or 2) add that code at a higher level.

        # TODO:  Do we need the flexibility to set steps_per_epoch, validation_steps, or validation_freq, or are there
        #  obvious and reasonable defaults to just use? w.r.t. validation steps and freq, I can only think of
        #  reasons to change them based on computational resource budgets.
        self.model.fit(
            train_features, train_responses, batch_size=self.network_config.batch_size,
            epochs=self.network_config.max_epochs, verbose=self.network_config.verbosity, callbacks=model_callbacks,
            validation_split=validation_split, validation_data=validation_data, shuffle=False,
            initial_epoch=self._get_initial_epoch(), steps_per_epoch=1, validation_steps=1, validation_freq=1
        )

    def fit_generator(
            self,
            train_generator: keras.utils.Sequence,
            validation_generator: keras.utils.Sequence = None,
            continue_training: bool = False,
    ) -> None:
        # TODO: I'm realizing we're kinda sorta putting in docstrings, but we could do them in such a way where we
        #  auto-generate documentation from the docstrings. I'd just need to look up the package and format for that.
        self._run_checks_before_model_fit(continue_training)
        if continue_training:
            self._set_model_initial_learning_rate_from_last_epoch()
        model_callbacks = callbacks.get_callbacks(self.network_config)

        # TODO:  Same parameter questions as with fit()
        # TODO:  Check whether psutil.cpu_count gives the right answer on SLURM, i.e., the number of CPUs available to
        #  the job and not the total number on the instance.
        self.model.fit_generator(
            train_generator, steps_per_epoch=1, epochs=self.network_config.max_epochs,
            verbose=self.network_config.verbosity, callbacks=model_callbacks, validation_data=validation_generator,
            validation_steps=1, validation_freq=1, max_queue_size=10, workers=psutil.cpu_count(logical=True),
            use_multiprocessing=True, shuffle=False, initial_epoch=self._get_initial_epoch(),
        )

    def _run_checks_before_model_fit(self, continue_training):
        if self.network_config.assert_gpu:
            utils.assert_gpu_available()

        assert not (continue_training is False and self._is_model_new is False), \
            'The parameter continue_training must be true to continue training an existing model'

    def _set_model_initial_learning_rate_from_last_epoch(self):
        K.set_value(self.model.optimizer.lr, self.history['lr'][-1])

    def _get_initial_epoch(self):
        return len(self.history.get('lr', list()))

    def predict(self, features):
        # TODO: Fabina, the verbosity below could be a config parameter, but you basically always want this off (it's either super
        # fast or we're running some structured read/write that has an external reporting setup)
        return self.model.predict(features, batch_size=self.network_config.batch_size, verbose=False)

    def predict_sequence(self, predict_sequence):
        # TODO:  reimplement if/when we need generators, ignore for now
        raise NotImplementedError
