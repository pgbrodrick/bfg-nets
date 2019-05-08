from rsCNN.configs import BaseConfigSection, ConfigOption


DEFAULT_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_OPTIMIZER = 'adam'
DEFAULT_PADDING = 'same'
DEFAULT_USE_BATCH_NORM = True
DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER = False


class BaseArchitectureOptions(BaseConfigSection):
    """
    Raw file configuration, information necessary to locate and parse the raw files.
    """
    filters = None
    inshape = None
    kernel_size = None
    n_classes = None
    optimizer = None
    output_activation = None
    padding = None
    use_batch_norm = None
    use_initial_colorspace_transformation_layer = None
    _config_options = [
        ConfigOption('filters', DEFAULT_FILTERS, int),
        ConfigOption('inshape', None, tuple),
        ConfigOption('kernel_size', DEFAULT_KERNEL_SIZE, tuple),
        ConfigOption('n_classes', None, int),
        ConfigOption('optimizer', DEFAULT_OPTIMIZER, str),
        ConfigOption('output_activation', None, str),
        ConfigOption('padding', DEFAULT_PADDING, str),
        ConfigOption('use_batch_norm', DEFAULT_USE_BATCH_NORM, bool),
        ConfigOption(
            'use_initial_colorspaced_transformation_layer', DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER, bool),
    ]

    def __init__(self):
        # We need to reorder field defaults given that child ArchitectureOptions will be adding field defaults
        self._config_options = sorted(self._config_options, key=lambda x: x.key)
        super().__init__()
