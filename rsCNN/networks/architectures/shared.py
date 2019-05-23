from rsCNN.configs import BaseConfigSection, ConfigOption


DEFAULT_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_PADDING = 'same'
DEFAULT_USE_BATCH_NORM = True
DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER = False


class BaseArchitectureOptions(BaseConfigSection):
    """Base class for architecture options, includes options that are generic to all architectures.
    """
    filters = None
    """int: Number of filters to use for initial convolutions, may increase in architectures that support the use_growth
    option."""
    inshape = None
    """tuple: The inshape of sample arrays passed to the model; e.g., 128x128x4 for a 128x128 image with four bands or
    channels."""
    kernel_size = None
    """tuple: The kernel size used for convolutions. Most often (3, 3) for a 3x3 kernel."""
    n_classes = None
    """int: The number of classes (filters) used in the final network layer. Example 1:  if the network is being trained
    to predict a single continuous variable, this should be 1. Example 2:  if the network is being trained to classify 
    pixels into five classes, this should be 5."""
    output_activation = None
    """str: The activation type for the final output layer. See Keras documentation for more details and available 
    options."""
    padding = None
    """str: See Keras documentation for more details and available options."""
    use_batch_norm = None
    """bool: Whether to use batch normalization layers. Currently only implemented after convolutions, but there's 
    evidence that it may be useful before convolutions in at least some architectures or applications, and we plan on 
    supporting both options in the near future."""
    use_initial_colorspace_transformation_layer = None
    """bool: Whether to use an initial colorspace transformation layer. There is evidence that model-learned color
    transformations can be more effective than other types of transformations."""

    _config_options = [
        ConfigOption('filters', DEFAULT_FILTERS, int),
        ConfigOption('inshape', None, tuple),
        ConfigOption('kernel_size', DEFAULT_KERNEL_SIZE, tuple),
        ConfigOption('n_classes', None, int),
        ConfigOption('output_activation', None, str),
        ConfigOption('padding', DEFAULT_PADDING, str),
        ConfigOption('use_batch_norm', DEFAULT_USE_BATCH_NORM, bool),
        ConfigOption(
            'use_initial_colorspace_transformation_layer', DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER, bool),
    ]
    _config_options_extra = list()

    def __init__(self):
        # We need to reorder option defaults given that child ArchitectureOptions will be adding option defaults
        self._config_options = self._config_options + self._config_options_extra
        self._config_options = sorted(self._config_options, key=lambda x: x.key)
        super().__init__()
