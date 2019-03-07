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

        # Optional arguments
        self.dir_out = kwargs.get('dir_out', './')
        self.filepath_model_out = kwargs.get('filepath_model_out', 'model.h5')
        self.filepath_history_out = kwargs.get('filepath_history_out', 'history.json')
        self.checkpoint_periods = kwargs.get('checkpoint_periods', 5)
        self.verbosity = kwargs.get('verbosity', 1)
        self.append_existing = kwargs.get('append_existing', False)

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
