import rsCNN.configs


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


class BaseArchitectureOptions(rsCNN.configs.BaseConfigSection):
    """Base class for architecture options, includes options that are generic to all architectures.
    """
    _filters_type = int
    filters = rsCNN.configs.DEFAULT_REQUIRED_VALUE
    """int: Number of filters to use for initial convolutions, may increase in architectures that support the use_growth
    option."""
    _inshape_type = tuple
    inshape = rsCNN.configs.DEFAULT_REQUIRED_VALUE
    """tuple: The inshape of sample arrays passed to the model; e.g., 128x128x4 for a 128x128 image with four bands or
    channels."""
    _kernel_size_type = tuple
    kernel_size = DEFAULT_KERNEL_SIZE
    """tuple: The kernel size used for convolutions. Most often (3, 3) for a 3x3 kernel."""
    _n_classes_type = int
    n_classes = rsCNN.configs.DEFAULT_REQUIRED_VALUE
    """int: The number of classes (filters) used in the final network layer. Example 1:  if the network is being trained
    to predict a single continuous variable, this should be 1. Example 2:  if the network is being trained to classify 
    pixels into five classes, this should be 5."""
    _output_activation_type = str
    output_activation = rsCNN.configs.DEFAULT_REQUIRED_VALUE
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
