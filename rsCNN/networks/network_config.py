import ast
import configparser
import os
from typing import Iterable

from rsCNN.networks import architectures


def read_network_config_from_file(filepath):
    config = configparser.ConfigParser()
    config.read(filepath)
    config_kwargs = dict()
    for section in config.sections():
        for key, value in section.items():
            assert key not in config_kwargs, 'Configuration file contains multiple entries for key:  {}'.format(key)
            # Note:  the following doesn't work with floats written as '10**-4' or strings without surrounding quotes
            value = ast.literal_eval(value)
            config_kwargs[key] = value
    return config_kwargs


class NetworkConfig(object):
    """ A wrapper class designed to hold all relevant configuration information for the
        training of a new network.
    """

    # TODO: maybe add the args to config template, which means we may just read in directly
    # TODO: typedef loss_function
    def __init__(self, network_type: str, loss_function, inshape: Iterable[int], n_classes: Iterable[int], **kwargs):
        """
          Arguments:
          network_type - str
            Style of the network to use.  Options are:
              flex_unet
              flat_regress_net
          loss_function - function
            Keras or tensor flow based loss function for the cnn.
          inshape - tuple/list
            Designates the input shape of an image to be passed to
            the network.
          n_classes - tuple/list
            Designates the output shape of targets to be fit by the network
        """
        self.network_type = network_type
        self.loss_function = loss_function
        self.inshape = inshape
        self.n_classes = n_classes
        architecture_creator = architectures.get_architecture_creator(self.network_type)
        self.create_model = architecture_creator.create_model
        self.architecture_options = architecture_creator.parse_architecture_options(**kwargs)

        # Training arguments
        self.batch_size = kwargs.get('batch_size', 1)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.n_noimprovement_repeats = kwargs.get('n_noimprovement_repeats', 10)
        self.verification_fold = kwargs.get('verification_fold', None)
        self.optimizer = kwargs.get('optimizer', 'adam')

        # Optional arguments
        self.dir_out = kwargs.get('dir_out', './')
        self.filename_model = os.path.join(self.dir_out, kwargs.get('filename_model', 'model.h5'))
        self.filename_history = os.path.join(self.dir_out, kwargs.get('filename_history', 'history.json'))

        # Model
        self.checkpoint_periods = kwargs.get('checkpoint_periods', 5)
        self.verbosity = kwargs.get('verbosity', 1)
        self.assert_gpu = kwargs.get('assert_gpu', False)
        # TODO:  unclear name, but could be streamlined? add to config template if we keep
        self.append_existing = kwargs.get('append_existing', False)

        # Callbacks
        self.callbacks_use_tensorboard = kwargs.get('callbacks_use_tensorboard', True)
        self.dirname_tensorboard = kwargs.get('dirname_tensorboard', 'tensorboard')
        self.tensorboard_update_freq = kwargs.get('tensorboard_update_freq', 'epoch')
        self.tensorboard_histogram_freq = kwargs.get('tensorboard_histogram_freq', 0)
        self.tensorboard_write_graph = kwargs.get('tensorboard_write_graph', True)
        self.tensorboard_write_grads = kwargs.get('tensorboard_write_grads', False)
        self.tensorboard_write_images = kwargs.get('tensorboard_write_images', True)

        self.callbacks_use_early_stopping = kwargs.get('callbacks_use_early_stopping', True)
        self.early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 10**-4)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 50)

        self.callbacks_use_reduced_learning_rate = kwargs.get('callbacks_use_reduced_learning_rate', True)
        self.reduced_learning_rate_factor = kwargs.get('reduced_learning_rate_factor', 0.5)
        self.reduced_learning_rate_min_delta = kwargs.get('reduced_learning_rate_min_delta', 10**-4)
        self.reduced_learning_rate_patience = kwargs.get('reduced_learning_rate_patience', 10)

        self.callbacks_use_terminate_on_nan = kwargs.get('callbacks_use_terminate_on_nan', True)
