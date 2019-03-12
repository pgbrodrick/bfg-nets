from typing import Iterable

import keras
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization


# TODO:  Convert to kwargs with default settings matching those of NetworkConfig? Would help by giving examples of
#  reasonable or expected values, allow use to call function easily without setting config (i.e., fewer params), allow
#  user to provide params in different order, etc.
def flat_regress_net(
    inshape: Iterable[int],
    n_classes: int,
    conv_depth: int,
    batch_norm: bool,
    n_layers: int,
    conv_pattern: Iterable[int],
    output_activation: keras.layers.Activation,
) -> keras.models.Model:
    """ Construct a flat style network with flexible shape

    Arguments:
    inshape - tuple/list
      Designates the input shape of an image to be passed to
      the network.
      # TODO:  Do we want to enforce a particular number of dimensions?
    n_classes - int
      The number of classes the network is meant to regress
    kwargs - dict
      A dictionary of optional keyword arguments, which may contain
      extra keywords.  Values to use are:
      conv_depth - int/str
        If integer, a fixed number of convolution filters to use
        in the network.  If 'growth' tells the network to grow
        in depth to maintain a constant number of neurons.
        # TODO:  Is growth the default if integers are not supplied? If so, maybe we say None == growth functionality?
      batch_norm - bool
        Whether or not to use batch normalization after each layer.
    # TODO:  info for conv_pattern, currently unclear

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
        # TODO:  Phil:  no batch norm after first convolution? just curious
        b1 = Conv2D(conv_depth, (kernel_size, kernel_size), activation='relu', padding='same')(b1)
        if (batch_norm):
            b1 = BatchNormalization()(b1)

    output_layer = Conv2D(n_classes, (1, 1), activation=output_activation, padding='same')(b1)
    model = keras.models.Model(input=inlayer, output=output_layer)
    return model
