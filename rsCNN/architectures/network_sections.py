from typing import Tuple


import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, ReLU, UpSampling2D


def colorspace_transformation(inshape: Tuple[int, int, int], inlayer: keras.layers,
                              batch_normalization: bool = False):
    """ Perform a series of layer transformations prior to the start of the main network.

    :argument Tuple(int, int, int) inshape: Shape of the incoming layer.
    :argument keras.layers inlayer: Input layer to the transformation.
    :argument Optional(bool) batch_normalization: Whether or not to use batch normalization.

    :return keras.layers outlayer: Keras layer ready to start the main network
    """
    intermediate_color_depth = int(inshape[-1] ** 2)
    conv = Conv2D(filters=intermediate_color_depth, kernel_size=(1, 1), padding='same')(inlayer)
    conv = Conv2D(filters=inshape[-1], kernel_size=(1, 1), padding='same')(conv)
    if (batch_normalization):
        conv = BatchNormalization()(conv)

    return conv


