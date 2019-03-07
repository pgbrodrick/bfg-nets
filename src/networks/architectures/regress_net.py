import keras

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

from src.networks.config import NetworkConfig


# TODO:  Phil:  like with u-net, is this the name we want to use? If so, please rename the script how you see fit.


class FlexUnetConfig(NetworkConfig):

    def __init__(self, network_type, inshape, n_classes, **kwargs):
        super().__init__(network_type, inshape, n_classes)
        self.conv_depth = kwargs.get('conv_depth', 16)
        self.batch_norm = kwargs.get('batch_norm', False)
        self.n_layers = kwargs.get('n_layers', 8)
        self.conv_pattern = kwargs.get('conv_pattern', [3])
        self.output_activation = kwargs.get('output_activation', 'softmax')


def flat_regress_net(inshape, n_classes, conv_depth, batch_norm, n_layers, conv_pattern, output_activation):
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
        if (n_layers % len(conv_pattern) != 0):
            Exception('conv_pattern must divide into n_layers')
            quit()

    if (isinstance(conv_depth, int) == False):
        Exception('conv_depth parameter must be an integer')
        quit()

    inlayer = keras.layers.Input(inshape)
    b1 = inlayer
    for i in range(n_layers):
        b1 = Conv2D(conv_depth, (conv_pattern[i % len(conv_pattern)], conv_pattern[i %
                                                                                   len(conv_pattern)]), activation='relu', padding='same')(b1)
        b1 = Conv2D(conv_depth, (conv_pattern[i % len(conv_pattern)], conv_pattern[i %
                                                                                   len(conv_pattern)]), activation='relu', padding='same')(b1)
        if (batch_norm):
            b1 = BatchNormalization()(b1)

    output_layer = Conv2D(n_classes, (1, 1), activation=output_activation, padding='same')(b1)
    model = keras.models.Model(input=inlayer, output=output_layer)
    return model

