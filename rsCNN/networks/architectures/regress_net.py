from typing import Tuple

import keras

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization


# TODO:  Convert to kwargs with default settings, use those default settings in NetworkConfig

def flat_regress_net(
    inshape: Tuple[int, int, int],
    n_classes: int,
    conv_depth: int,
    batch_norm: bool,
    n_layers: int,
    conv_pattern: Tuple[int],
    output_activation: str,
) -> keras.models.Model:
    """ Construct a flat style network with flexible shape

    Arguments:
    inshape - tuple/list
      Designates the input shape of an image to be passed to
      the network.
    n_classes - int
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
      batch_norm - bool
        Whether or not to use batch normalization after each layer.

    Returns:
      A flexible flat style network keras network.
    """
    if (len(conv_pattern) > 0):
        assert (n_layers % len(conv_pattern) == 0), 'conv_pattern must divide into n_layers'

    inlayer = keras.layers.Input(inshape)
    b1 = inlayer
    for i in range(n_layers):
        kernel_size = conv_pattern[i % len(conv_pattern)]
        b1 = Conv2D(conv_depth, (kernel_size, kernel_size), activation='relu', padding='same')(b1)
        b1 = Conv2D(conv_depth, (kernel_size, kernel_size), activation='relu', padding='same')(b1)
        if (batch_norm):
            b1 = BatchNormalization()(b1)

    output_layer = Conv2D(n_classes, (1, 1), activation=output_activation, padding='same')(b1)
    model = keras.models.Model(inputs=inlayer, outputs=output_layer)
    return model
