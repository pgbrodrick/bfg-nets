from typing import List, Tuple, Union

import keras

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization


DEFAULT_INITIAL_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_NUM_LAYERS = 8
DEFAULT_PADDING = 'same'
DEFAULT_USE_BATCH_NORM = True


def parse_architecture_options(**kwargs):
    return {
        'initial_filters': kwargs.get('initial_filters', DEFAULT_INITIAL_FILTERS),
        'kernel_size': kwargs.get('kernel_size', DEFAULT_KERNEL_SIZE),
        'num_layers': kwargs.get('num_layers', DEFAULT_NUM_LAYERS),
        'padding': kwargs.get('padding', DEFAULT_PADDING),
        'use_batch_norm': kwargs.get('use_batch_norm', DEFAULT_USE_BATCH_NORM),
    }


def create_model(
        input_shape: Tuple[int, int, int],
        num_outputs: int,
        output_activation: str,
        initial_filters: int = DEFAULT_INITIAL_FILTERS,
        kernel_size: Union[Tuple[int, int], List[Tuple[int, int]]] = DEFAULT_KERNEL_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        padding: str = DEFAULT_PADDING,
        use_batch_norm: bool = DEFAULT_USE_BATCH_NORM,
) -> keras.models.Model:
    """ Construct a flat style network with flexible shape

    Arguments:
    input_shape - tuple/list
      Designates the input shape of an image to be passed to
      the network.
    num_outputs - int
      The number of classes the network is meant to regress
    kwargs - dict
      A dictionary of optional keyword arguments, which may contain
      extra keywords.  Values to use are:

      conv_pattern - tuple/list
        Designates the (repeating) order of convolution filter sizes.
      conv_depth - int/str
        If integer, a fixed number of convolution filters to use
        in the network.  If 'growth' tells the network to grow
        in depth to maintain a constant number of neurons.
      use_batch_norm - bool
        Whether or not to use batch normalization after each layer.

    Returns:
      A flexible flat style network keras network.
    """
    inlayer = keras.layers.Input(input_shape)

    if type(kernel_size) is tuple:
        kernel_sizes = [kernel_size] * len(num_layers)
    else:
        assert len(kernel_size) == num_layers, 'If providing a list of kernel sizes, length must equal num_layers'
        kernel_sizes = kernel_size

    b1 = inlayer
    for kernel_size in kernel_sizes:
        b1 = Conv2D(filters=initial_filters, kernel_size=kernel_size, padding=padding)(b1)
        b1 = Conv2D(filters=initial_filters, kernel_size=kernel_size, padding=padding)(b1)
        if use_batch_norm:
            b1 = BatchNormalization()(b1)

    output_layer = Conv2D(num_outputs, (1, 1), activation=output_activation, padding=padding)(b1)
    model = keras.models.Model(inputs=inlayer, outputs=output_layer)
    return model
