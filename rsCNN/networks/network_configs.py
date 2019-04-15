import ast
from collections import OrderedDict
import configparser
import os
from typing import Tuple

from rsCNN.networks import architectures


FILENAME_NETWORK_CONFIG = 'network_config.ini'


def load_network_config(dir_config, filename: str = None) -> OrderedDict:
    if filename is None:
        filename = FILENAME_NETWORK_CONFIG
    config = configparser.ConfigParser()
    config.read(os.path.join(dir_config, filename))
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
            except:
                value = str(value)
            kwargs[key] = value
    return create_network_config(**kwargs)


def save_network_config(network_config: dict, dir_config: str, filename: str = None) -> None:
    if not filename:
        filename = FILENAME_NETWORK_CONFIG
    writer = configparser.ConfigParser()
    for section, section_items in network_config.items():
        writer[section] = {}
        for key, value in section_items.items():
            if (key == 'create_model'):
                continue
            writer[section][key] = str(value)
        #writer[section] = section_items
    with open(os.path.join(dir_config, filename), 'w') as file_:
        writer.write(file_)


def create_network_config(
        architecture: str,
        inshape: Tuple[int, int, int],
        n_classes: int,
        loss_metric: str,
        output_activation: str,
        **kwargs
) -> OrderedDict:
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
    config = OrderedDict()

    config['model'] = {
        'dir_out': kwargs.get('dir_out'),
        'verbosity': kwargs.get('verbosity', 1),
        'assert_gpu': kwargs.get('assert_gpu', False),
    }

    architecture_creator = architectures.get_architecture_creator(architecture)
    config['architecture'] = {
        'architecture': architecture,
        'inshape': inshape,
        'internal_window_radius': kwargs.get('internal_window_radius', int(inshape[0]/2.)),
        'n_classes': n_classes,
        'loss_metric': loss_metric,
        'create_model': architecture_creator.create_model,
        'weighted': kwargs.get('weighted', False),
    }

    config['architecture_options'] = architecture_creator.parse_architecture_options(**kwargs)
    config['architecture_options']['output_activation'] = output_activation

    config['training'] = {
        'apply_random_transformations': kwargs.get('apply_random_transformations', False),
        'max_epochs': kwargs.get('max_epochs', 100),
        'optimizer': kwargs.get('optimizer', 'adam'),
        'batch_size': kwargs.get('batch_size', 100),
        'modelfit_nan_value': kwargs.get('modelfit_nan_value', -100),
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

    config['callbacks_reduced_learning_rate'] = {
        'use_reduced_learning_rate': kwargs.get('use_reduced_learning_rate', True),
        'rlr_factor': kwargs.get('rlr_factor', 0.5),
        'rlr_min_delta': kwargs.get('rlr_min_delta', 0.0001),
        'rlr_patience': kwargs.get('rlr_patience', 10),
    }
    return config


def compare_network_configs_get_differing_items(config_a, config_b):
    differing_items = list()
    all_sections = set(list(config_a.keys()) + list(config_b.keys()))
    for section in all_sections:
        section_a = config_a.get(section, dict())
        section_b = config_b.get(section, dict())
        all_keys = set(list(section_a.keys()) + list(section_b.keys()))
        for key in all_keys:
            value_a = section_a.get(key, None)
            value_b = section_b.get(key, None)
            if value_a != value_b:
                differing_items.append((section, key, value_a, value_b))
    return differing_items
