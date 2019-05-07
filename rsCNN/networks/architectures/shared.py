from rsCNN.configs import BaseConfigSection


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
    # TODO:  document
    _field_defaults = [
        ('filters', DEFAULT_FILTERS, int),
        ('inshape', None, tuple),
        ('kernel_size', DEFAULT_KERNEL_SIZE, tuple),
        ('n_classes', None, int),
        ('optimizer', DEFAULT_OPTIMIZER, str),
        ('output_activation', None, str),
        ('padding', DEFAULT_PADDING, str),
        ('use_batch_norm', DEFAULT_USE_BATCH_NORM, bool),
        ('use_initial_colorspaced_transformation_layer', DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER, bool),
    ]

    def __init__(self):
        # We need to reorder field defaults given that child ArchitectureOptions will be adding field defaults
        self._field_defaults = sorted(self._field_defaults)
        super().__init__()
