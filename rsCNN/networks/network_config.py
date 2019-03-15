import ast
import configparser
import os
from typing import Tuple

from rsCNN.networks import architectures
from rsCNN.utils import DIR_TEMPLATES


def create_config_files_from_network_defaults():
    value_required = 'REQUIRED'
    for architecture in ('regress_net', 'residual_net', 'residual_unet', 'unet'):
        config = create_network_config(
            architecture=architecture, model_name=value_required, inshape=(0, 0, 0),
            n_classes=0, loss_function=value_required, output_activation=value_required,
        )
        config['architecture'].pop('create_model')
        with open(os.path.join(DIR_TEMPLATES, architecture + '.ini'), 'w') as file_:
            config.write(file_)


def read_network_config_from_file(filepath):
    config = configparser.ConfigParser()
    config.read(filepath)
    kwargs = dict()
    for section in config.sections():
        for key, value in config[section].items():
            assert key not in kwargs, 'Configuration file contains multiple entries for key:  {}'.format(key)
            # Note:  the following doesn't work with floats written as '10**-4' or strings without surrounding quotes
            value = ast.literal_eval(value)
            kwargs[key] = value
    return create_network_config(**kwargs)


def create_network_config(
        architecture: str,
        model_name: str,
        inshape: Tuple[int, int, int],
        n_classes: int,
        loss_function: str,
        output_activation: str,
        **kwargs
) -> configparser.ConfigParser:
    """
      Arguments:
      architecture - str
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
    config = configparser.ConfigParser()

    config['model'] = {
        'model_name': model_name,
        'dir_out': os.path.join(kwargs.get('dir_out', './'), model_name),
        'verbosity': kwargs.get('verbosity', 1),
        'assert_gpu': kwargs.get('assert_gpu', False),
    }

    architecture_creator = architectures.get_architecture_creator(architecture)

    config['architecture'] = {
        'architecture': architecture,
        'inshape': inshape,
        'n_classes': n_classes,
        'loss_function': loss_function,
        'output_activation': output_activation,
        'create_model': architecture_creator.create_model,
    }

    config['architecture_options'] = architecture_creator.parse_architecture_options(**kwargs)

    config['training'] = {
        'batch_size': kwargs.get('batch_size', 1),
        'max_epochs': kwargs.get('max_epochs', 100),
        'optimizer': kwargs.get('optimizer', 'adam'),
    }

    # TODO:  handle datetime addition to tensorboard directory name
    config['callbacks_general'] = {
        'checkpoint_periods': kwargs.get('checkpoint_periods', 5),
        'callbacks_use_terminate_on_nan': kwargs.get('callbacks_use_terminate_on_nan', True),
    }

    config['callbacks_tensorboard'] = {
        'callbacks_use_tensorboard': kwargs.get('callbacks_use_tensorboard', True),
        'dirname_prefix_tensorboard': kwargs.get('dirname_prefix_tensorboard', 'tensorboard'),
        'tensorboard_update_freq': kwargs.get('tensorboard_update_freq', 'epoch'),
        'tensorboard_histogram_freq': kwargs.get('tensorboard_histogram_freq', 0),
        'tensorboard_write_graph': kwargs.get('tensorboard_write_graph', True),
        'tensorboard_write_grads': kwargs.get('tensorboard_write_grads', False),
        'tensorboard_write_images': kwargs.get('tensorboard_write_images', True),
    }

    config['callbacks_early_stopping'] = {
        'callbacks_use_early_stopping': kwargs.get('callbacks_use_early_stopping', True),
        'early_stopping_min_delta': kwargs.get('early_stopping_min_delta', 0.0001),
        'early_stopping_patience': kwargs.get('early_stopping_patience', 50),
    }

    config['callbacks_reduced_learning_rate'] = {
        'callbacks_use_reduced_learning_rate': kwargs.get('callbacks_use_reduced_learning_rate', True),
        'reduced_learning_rate_factor': kwargs.get('reduced_learning_rate_factor', 0.5),
        'reduced_learning_rate_min_delta': kwargs.get('reduced_learning_rate_min_delta', 0.0001),
        'reduced_learning_rate_patience': kwargs.get('reduced_learning_rate_patience', 10),
    }
    return config
