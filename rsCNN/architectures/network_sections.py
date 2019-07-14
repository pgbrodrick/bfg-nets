from typing import Tuple


import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, ReLU, UpSampling2D


def colorspace_transformation(inshape: Tuple[int, int, int], inlayer: keras.layers,
                              batch_normalization: bool = False):
    """ Perform a series of layer transformations prior to the start of the main network.

    :argument Tuple(int, int, int) inshape: Shape of the incoming layer.
    :argument keras.layers inlayer: Input layer to the transformation.
    :argument Optional(bool) batch_normalization: Whether or not to use batch normalization.

    :return keras.layers output_layer: Keras layer ready to start the main network
    """
    intermediate_color_depth = int(inshape[-1] ** 2)
    output_layer = Conv2D(filters=intermediate_color_depth, kernel_size=(1, 1), padding='same')(inlayer)
    output_layer = Conv2D(filters=inshape[-1], kernel_size=(1, 1), padding='same')(output_layer)
    if (batch_normalization):
        output_layer = BatchNormalization()(output_layer)

    return output_layer


def Conv2D_Options(inlayer: keras.layers, options: dict):
    """ Perform a keras 2D convolution with the specified options.

    :argument keras.layers inlayer: Input layer to the convolution.
    :argument dict options: All options to pass into the input layer.

    :return keras.layers.Conv2D output_layer: Keras layer ready to start the main network
    """

    use_batch_norm = options.pop('use_batch_norm',False)
    output_layer = Conv2D(**options)(inlayer)
    if use_batch_norm:
        output_layer = BatchNormalization()(output_layer)

    return output_layer
