from typing import Tuple

import keras


DEFAULT_BLOCK_STRUCTURE = (2, 2, 2, 2)
DEFAULT_BATCH_NORM = True
DEFAULT_INITIAL_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_PADDING = 'same'
DEFAULT_POOL_SIZE = (3, 3)
DEFAULT_STRIDES = (1, 1)


def parse_architecture_options(**kwargs):
    return {
        'block_structure': kwargs.get('block_structure', DEFAULT_BLOCK_STRUCTURE),
        'batch_norm': kwargs.get('batch_norm', DEFAULT_BATCH_NORM),
        'initial_filters': kwargs.get('initial_filters', DEFAULT_INITIAL_FILTERS),
        'kernel_size': kwargs.get('kernel_size', DEFAULT_KERNEL_SIZE),
        'padding': kwargs.get('padding', DEFAULT_PADDING),
        'output_activation': kwargs['output_activation'],
    }


def create_model(
    input_shape: Tuple[int, int, int],
    num_outputs: int,
    output_activation: str,
    block_structure: Tuple[int, ...] = DEFAULT_BLOCK_STRUCTURE,
    batch_norm: bool = DEFAULT_BATCH_NORM,
    initial_filters: int = DEFAULT_INITIAL_FILTERS,
    kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
    padding: str = DEFAULT_PADDING,
) -> keras.models.Model:

    # Initial convolution
    input_layer = keras.layers.Input(shape=input_shape)
    conv = keras.layers.Conv2D(filters=initial_filters, kernel_size=kernel_size, padding=padding)(input_layer)

    # Iterate blocks and subblocks
    subblock_input = conv
    filters = initial_filters
    for idx_block, num_subblocks in enumerate(block_structure):
        for idx_sublayer in range(num_subblocks):
            subblock = subblock_input
            if batch_norm:
                subblock = keras.layers.BatchNormalization()(subblock)
            subblock = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(subblock)
            subblock_input = _add_residual_shortcut(subblock_input, subblock)
        filters *= 2

    # Output convolutions
    output_layer = subblock_input
    if batch_norm:
        output_layer = keras.layers.BatchNormalization()(output_layer)
    output_layer = keras.layers.Conv2D(
        filters=num_outputs, kernel_size=(1, 1), padding='same', activation=output_activation)(output_layer)
    return keras.models.Model(inputs=[input_layer], outputs=[output_layer])


def _add_residual_shortcut(input_layer: keras.layers.Layer, residual_module: keras.layers.Layer):
    """
    Adds a shortcut connection by combining a input tensor and residual module
    """
    shortcut = input_layer

    # We need to apply a convolution if the input and block shapes do not match, every block transition
    input_shape = keras.backend.int_shape(input_layer)[1:]
    residual_shape = keras.backend.int_shape(residual_module)[1:]
    if input_shape != residual_shape:
        strides = (int(round(input_shape[0] / residual_shape[0])), int(round(input_shape[1] / residual_shape[1])))
        shortcut = keras.layers.Conv2D(
            filters=residual_shape[-1], kernel_size=(1, 1), padding='valid', strides=strides)(shortcut)

    return keras.layers.add([shortcut, residual_module])
