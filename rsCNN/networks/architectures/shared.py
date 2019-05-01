from rsCNN.configs import BaseConfigSection


class BaseArchitectureOptions(BaseConfigSection):
    """
    Raw file configuration, information necessary to locate and parse the raw files.
    """
    filters = None
    kernel_size = None
    num_layers = None
    padding = None
    use_batch_norm = None
    use_initial_colorspace_transformation_layer = None
    # TODO:  document
    _field_defaults = [
        ('filters', 64, int),
        ('kernel_size', (3, 3), tuple),
        ('num_layers', 8, int),
        ('padding', 'same', str),
        ('use_batch_norm', True, bool),
        ('use_initial_colorspaced_transformation_layer', False, bool),
    ]
