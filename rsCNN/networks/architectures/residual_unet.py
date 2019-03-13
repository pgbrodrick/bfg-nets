from typing import Tuple

import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, MaxPooling2D, UpSampling2D


DEFAULT_BLOCK_STRUCTURE = (2, 2, 2, 2)
DEFAULT_BATCH_NORM = True
DEFAULT_INITIAL_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_PADDING = 'same'
DEFAULT_POOL_SIZE = (2, 2)
DEFAULT_STRIDES = (1, 1)


def parse_architecture_options(**kwargs):
    return {
        'block_structure': kwargs.get('block_structure', DEFAULT_BLOCK_STRUCTURE),
        'batch_norm': kwargs.get('batch_norm', DEFAULT_BATCH_NORM),
        'initial_filters': kwargs.get('initial_filters', DEFAULT_INITIAL_FILTERS),
        'kernel_size': kwargs.get('kernel_size', DEFAULT_KERNEL_SIZE),
        'padding': kwargs.get('padding', DEFAULT_PADDING),
        'pool_size': kwargs.get('pool_size', DEFAULT_POOL_SIZE),
        'strides': kwargs.get('strides', DEFAULT_STRIDES),
    }


def create_residual_network(
        input_shape: Tuple[int, int, int],
        num_outputs: int,
        output_activation: str,
        block_structure: Tuple[int, ...] = DEFAULT_BLOCK_STRUCTURE,
        batch_norm: bool = DEFAULT_BATCH_NORM,
        initial_filters: int = DEFAULT_INITIAL_FILTERS,
        kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
        padding: str = DEFAULT_PADDING,
        pool_size: Tuple[int, int] = DEFAULT_POOL_SIZE,
        strides: Tuple[int, int] = DEFAULT_STRIDES,
        use_growth: bool = False,
) -> keras.models.Model:

    # TODO:  unnest one layer, calculate that input_width stays above 8 for block structure, assert in beginning
    #assert min_input_width > 8

    # Need to track the following throughout the model creation
    input_width = input_shape[0]
    filters = initial_filters
    layers_pass_through = list()

    # Encodings
    input_layer = keras.layers.Input(shape=input_shape)
    encoder = input_layer
    # Each encoder block has a number of subblocks
    for num_subblocks in block_structure:
        for idx_sublayer in range(num_subblocks):
            # Store the subblock input for the residual connection
            input_subblock = encoder
            # Each subblock has two convolutions
            encoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(encoder)
            if (batch_norm):
                encoder = BatchNormalization()(encoder)
            encoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(encoder)
            if (batch_norm):
                encoder = BatchNormalization()(encoder)
            # Add the residual connection from the previous subblock output to the current subblock output
            encoder = _add_residual_shortcut(input_subblock, encoder)
        # Each encoder block passes its pre-pooled layers through to the decoder
        layers_pass_through.append(encoder)
        encoder = MaxPooling2D(pool_size=pool_size)(encoder)
        if use_growth:
            filters *= 2

    # Decodings
    decoder = encoder
    # Each decoder block has a number of subblocks, but in reverse order of encoder
    for num_subblocks, layer_passed_through in zip(reversed(block_structure), reversed(layers_pass_through)):
        for idx_sublayer in range(num_subblocks):
            # Store the subblock input for the residual connection
            input_subblock = decoder
            # Each subblock has two convolutions
            decoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(decoder)
            if (batch_norm):
                decoder = BatchNormalization()(decoder)
            decoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(decoder)
            if (batch_norm):
                decoder = BatchNormalization()(decoder)
            # Add the residual connection from the previous subblock output to the current subblock output
            decoder = _add_residual_shortcut(input_subblock, decoder)
        decoder = UpSampling2D(size=pool_size)(decoder)
        decoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(decoder)
        if (batch_norm):
            decoder = BatchNormalization()(decoder)
        decoder = Concatenate()([layer_passed_through, decoder])
        if use_growth:
            filters /= 2

    # Last convolutions
    output_layer = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(decoder)
    if (batch_norm):
        output_layer = BatchNormalization()(output_layer)
    output_layer = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(output_layer)
    if (batch_norm):
        output_layer = BatchNormalization()(output_layer)
    output_layer = Conv2D(
        filters=num_outputs, kernel_size=(1, 1), padding='same', activation=output_activation)(output_layer)
    return keras.models.Model(inputs=[input_layer], outputs=[output_layer])


def _add_residual_shortcut(input_tensor: keras.layers.Layer, residual_module: keras.layers.Layer):
    """
    Adds a shortcut connection by combining a input tensor and residual module
    """
    shortcut = input_tensor

    # We need to apply a convolution if the input and block shapes do not match, every block transition
    input_shape = keras.backend.int_shape(input_tensor)[1:]
    residual_shape = keras.backend.int_shape(residual_module)[1:]
    if input_shape != residual_shape:
        strides = (int(round(input_shape[0] / residual_shape[0])), int(round(input_shape[1] / residual_shape[1])))
        shortcut = keras.layers.Conv2D(
            filters=residual_shape[-1], kernel_size=(1, 1), padding='valid', strides=strides)(shortcut)

    return keras.layers.add([shortcut, residual_module])
