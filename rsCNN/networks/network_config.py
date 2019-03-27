import ast
import collections
import configparser
import os
from typing import Tuple

from rsCNN.networks import architectures
from rsCNN.utils import DIR_TEMPLATES


def create_config_files_from_network_defaults():
    value_required = 'REQUIRED'
    for architecture in ('change_detection', 'regress_net', 'residual_net', 'residual_unet', 'unet'):
        config = create_network_config(
            architecture=architecture, model_name=value_required, inshape=(0, 0, 0),
            n_classes=0, loss_metric=value_required, output_activation=value_required,
        )
        config['architecture'].pop('create_model')
        writer = configparser.ConfigParser()
        for section, section_items in config.items():
            writer[section] = section_items
        with open(os.path.join(DIR_TEMPLATES, architecture + '.ini'), 'w') as file_:
            writer.write(file_)


def read_network_config_from_file(filepath):
    config = configparser.ConfigParser()
    config.read(filepath)
    kwargs = dict()
    for section in config.sections():
        for key, value in config[section].items():
            assert key not in kwargs, 'Configuration file contains multiple entries for key:  {}'.format(key)
            # Note:  literal_eval doesn't work with scientific notation '10**-4' or strings without quotes. The
            # try/except catches string errors which are very inconvenient to address in the config files with quotes,
            # but the float issue isn't a problem if we're just careful. There's not an out-of-the-box way to sanitize
            # everything, unfortunately, so just be diligent with config files.
            try:
                value = ast.literal_eval(value)
            except ValueError:
                value = str(value)
            kwargs[key] = value
    return create_network_config(**kwargs)


def create_network_config(
        architecture: str,
        model_name: str,
        inshape: Tuple[int, int, int],
        n_classes: int,
        loss_metric: str,
        output_activation: str,
        **kwargs
) -> collections.OrderedDict:
    """
      Arguments:
      architecture - str
        Style of the network to use.  Options are:
          flex_unet
          flat_regress_net
      loss_metric - str
        Style of loss function to implement.
      inshape - tuple/list
        Designates the input shape of an image to be passed to
        the network.
      n_classes - tuple/list
        Designates the output shape of targets to be fit by the network
    """
    config = collections.OrderedDict()

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
        'loss_metric': loss_metric,
        'create_model': architecture_creator.create_model,
        'weighted': kwargs.get('weighted', False),
    }

    config['architecture_options'] = architecture_creator.parse_architecture_options(**kwargs)
    config['architecture_options']['output_activation'] = output_activation

    config['training'] = {
        'apply_random_transformations': kwargs.get('apply_random_transformations', False),
        'batch_size': kwargs.get('batch_size', 1),
        'max_epochs': kwargs.get('max_epochs', 100),
        'optimizer': kwargs.get('optimizer', 'adam'),
    }

    config['callbacks_general'] = {
        'checkpoint_periods': kwargs.get('checkpoint_periods', 5),
        'use_terminate_on_nan': kwargs.get('use_terminate_on_nan', True),
    }

    config['callbacks_tensorboard'] = {
        'use_tensorboard': kwargs.get('use_tensorboard', True),
        'dirname_prefix_tensorboard': kwargs.get('dirname_prefix_tensorboard', 'tensorboard'),
        't_update_freq': kwargs.get('t_update_freq', 'epoch'),
        't_histogram_freq': kwargs.get('t_histogram_freq', 0),
        't_write_graph': kwargs.get('t_write_graph', True),
        't_write_grads': kwargs.get('t_write_grads', False),
        't_write_images': kwargs.get('t_write_images', True),
    }

    config['callbacks_early_stopping'] = {
        'use_early_stopping': kwargs.get('use_early_stopping', True),
        'es_min_delta': kwargs.get('es_min_delta', 0.0001),
        'es_patience': kwargs.get('es_patience', 50),
    }

    config['callbacks_learning_rate_scheduler'] = {
        'use_learning_rate_scheduler': kwargs.get('use_learning_rate_scheduler', False),
        'lrs_use_linear': kwargs.get('lrs_use_linear', True),
        'lrs_use_decay': kwargs.get('lrs_use_decay', False),
        'lrs_rate_linear': kwargs.get('lrs_rate_linear', 0.0001),
        'lrs_rate_decay': kwargs.get('lrs_rate_decay', 0.99),
        'lrs_minimum': kwargs.get('lrs_minimum', 0.0000001),
    }

    config['callbacks_reduced_learning_rate'] = {
        'use_reduced_learning_rate': kwargs.get('use_reduced_learning_rate', True),
        'rlr_factor': kwargs.get('rlr_factor', 0.5),
        'rlr_min_delta': kwargs.get('rlr_min_delta', 0.0001),
        'rlr_patience': kwargs.get('rlr_patience', 10),
    }
    return config
