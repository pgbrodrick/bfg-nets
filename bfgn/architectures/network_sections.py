from typing import Tuple

import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D

# TODO: rename functions where necessary


def colorspace_transformation(
    inshape: Tuple[int, int, int], inlayer: keras.layers, batch_normalization: bool = False
) -> keras.layers:
    """ Perform a series of layer transformations prior to the start of the main network.

    Args:
        inshape: Shape of the incoming layer.
        inlayer: Input layer to the transformation.
        batch_normalization: Whether or not to use batch normalization.

    Returns:
        output_layer: Keras layer ready to start the main network
    """
    intermediate_color_depth = int(inshape[-1] ** 2)
    output_layer = Conv2D(filters=intermediate_color_depth, kernel_size=(1, 1), padding="same")(inlayer)
    output_layer = Conv2D(filters=inshape[-1], kernel_size=(1, 1), padding="same")(output_layer)
    if batch_normalization:
        output_layer = BatchNormalization()(output_layer)

    return output_layer


def Conv2D_Options(inlayer: keras.layers, options: dict) -> keras.layers.Conv2D:
    """ Perform a keras 2D convolution with the specified options.

    Args:
        inlayer: Input layer to the convolution.
        options: All options to pass into the input layer.

    Returns:
        output_layer: Keras layer ready to start the main network
    """

    use_batch_norm = options.pop("use_batch_norm", False)
    output_layer = Conv2D(**options)(inlayer)
    if use_batch_norm:
        output_layer = BatchNormalization()(output_layer)

    return output_layer


def dense_2d_block(inlayer: keras.layers, conv_options: dict, block_depth: int) -> keras.layers.Conv2D:
    """ Create a single, dense block.

    Args:
        inlayer: Input layer to the convolution.
        conv_options: All options to pass into the input convolution layer.
        block_depth: How deep (many layers) is the dense block.

    Returns:
        output_layer: Keras layer ready to start the main network
    """

    concat_layer = inlayer
    for _block_step in range(block_depth):
        # 1x1 conv on inputs to layer
        dense_layer = \
            Conv2D(conv_options['filters'], kernel_size=1, activation=conv_options['activation'])(concat_layer)
        # kxk conv
        dense_layer = Conv2D_Options(dense_layer, conv_options)
        # concatenate
        concat_layer = Concatenate(axis=-1)([concat_layer, dense_layer])

    return dense_layer
