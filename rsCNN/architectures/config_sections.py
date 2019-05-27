import importlib
from types import ModuleType

import keras

from rsCNN.configuration import DEFAULT_REQUIRED_VALUE, sections


# Global parameters
DEFAULT_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_PADDING = 'same'
DEFAULT_USE_BATCH_NORM = False
DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER = False

# Architecture-specific parameters
DEFAULT_BLOCK_STRUCTURE = (2, 2, 2, 2)
DEFAULT_DILATION_RATE = 2
DEFAULT_MIN_CONV_WIDTH = 8
DEFAULT_NUM_LAYERS = 8
DEFAULT_POOL_SIZE = (2, 2)
DEFAULT_USE_GROWTH = False


class BaseArchitectureConfigSection(sections.BaseConfigSection):
    """Base class for architecture config section, includes options that are generic to all architectures.
    """
    _filters_type = int
    filters = DEFAULT_REQUIRED_VALUE
    """int: Number of filters to use for initial convolutions, may increase in architectures that support the use_growth
    option."""
    _inshape_type = tuple
    inshape = DEFAULT_REQUIRED_VALUE
    """tuple: The inshape of sample arrays passed to the model; e.g., 128x128x4 for a 128x128 image with four bands or
    channels."""
    _kernel_size_type = tuple
    kernel_size = DEFAULT_KERNEL_SIZE
    """tuple: The kernel size used for convolutions. Most often (3, 3) for a 3x3 kernel."""
    _n_classes_type = int
    n_classes = DEFAULT_REQUIRED_VALUE
    """int: The number of classes (filters) used in the final network layer. Example 1:  if the network is being trained
    to predict a single continuous variable, this should be 1. Example 2:  if the network is being trained to classify 
    pixels into five classes, this should be 5."""
    _output_activation_type = str
    output_activation = DEFAULT_REQUIRED_VALUE
    """str: The activation type for the final output layer. See Keras documentation for more details and available 
    options."""
    _padding_type = str
    padding = DEFAULT_PADDING
    """str: See Keras documentation for more details and available options."""
    _use_batch_norm_type = bool
    use_batch_norm = DEFAULT_USE_BATCH_NORM
    """bool: Whether to use batch normalization layers. Currently only implemented after convolutions, but there's 
    evidence that it may be useful before convolutions in at least some architectures or applications, and we plan on 
    supporting both options in the near future."""
    _use_initial_colorspace_transformation_layer_type = bool
    use_initial_colorspace_transformation_layer = DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER
    """bool: Whether to use an initial colorspace transformation layer. There is evidence that model-learned color
    transformations can be more effective than other types of transformations."""

    def get_option_keys(self):
        # TODO:  figure out why the inherited get_option_keys is not returning anything. Maybe because of mixins?
        return [key for key in dir(self) if not key.startswith('_') and not callable(getattr(self, key))]


# TODO:  documentation

class AutoencoderMixin(object):
    _min_conv_width_type = int
    min_conv_width = DEFAULT_MIN_CONV_WIDTH
    _pool_size_type = tuple
    pool_size = DEFAULT_POOL_SIZE


class BlockMixin(object):
    _block_structure_type = tuple
    block_structure = DEFAULT_BLOCK_STRUCTURE


class DilationMixin(object):
    _dilation_rate_type = int
    dilation_rate = DEFAULT_DILATION_RATE


class FlatMixin(object):
    _num_layers_type = int
    num_layers = DEFAULT_NUM_LAYERS


class GrowthMixin(object):
    _use_growth_type = bool
    use_growth = DEFAULT_USE_GROWTH


def create_model_from_architecture_config_section(
        architecture_name: str,
        architecture_config_section: BaseArchitectureConfigSection
) -> keras.models.Model:
    """Creates a Keras model for a specific architecture using the provided options.

    # TODO:  figure out how to populate names automatically
    Args:
        architecture_name: Architecture to create. Currently available architectures are:  alex_net, dense_flat_net,
        dense_unet, dilation_net, flat_net, residual_dilation_net, residual_flat_net, residual_unet, and unet.
        architecture_config_section: Options for the specified architecture.

    Returns:
        Keras model object.
    """
    architecture_module = _import_architecture_module(architecture_name)
    kwargs = {key: getattr(architecture_config_section, key) for key in architecture_config_section.get_option_keys()}
    return architecture_module.create_model(**kwargs)


def get_architecture_config_section(architecture_name: str) -> BaseArchitectureConfigSection:
    """Gets architecture options for the specified architecture.

    # TODO:  figure out how to populate names automatically
    Args:
        architecture_name: Architecture to create. Currently available architectures are:  alex_net, dense_flat_net,
        dense_unet, dilation_net, flat_net, residual_dilation_net, residual_flat_net, residual_unet, and unet.

    Returns:
        Options for the specified architecture.
    """
    architecture_module = _import_architecture_module(architecture_name)
    return architecture_module.ArchitectureConfigSection()


def _import_architecture_module(architecture_name: str) -> ModuleType:
    try:
        architecture_module = importlib.import_module('rsCNN.architectures.{}'.format(architecture_name))
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Architecture {} is not a valid architecture'.format(architecture_name))
    return architecture_module
