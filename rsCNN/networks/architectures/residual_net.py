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
    }


def create_residual_network(
    input_shape: Tuple[int, int, int],
    num_outputs: int,
    block_structure: Tuple[int, ...] = DEFAULT_BLOCK_STRUCTURE,
    batch_norm: bool = DEFAULT_BATCH_NORM,
    initial_filters: int = DEFAULT_INITIAL_FILTERS,
    kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
    padding: str = DEFAULT_PADDING,
) -> keras.models.Model:

    # Initial convolution
    input_tensor = keras.layers.Input(shape=input_shape)
    conv = keras.layers.Conv2D(filters=initial_filters, kernel_size=kernel_size, padding=padding)(input_tensor)
    if batch_norm:
        conv = keras.layers.BatchNormalization()(conv)

    # Iterate blocks and subblocks
    subblock_input = conv
    filters = initial_filters
    for idx_block, num_subblocks in enumerate(block_structure):
        for idx_sublayer in range(num_subblocks):
            is_first_subblock_in_first_block = idx_block == 0 and idx_sublayer == 0
            subblock = subblock_input
            if batch_norm and not is_first_subblock_in_first_block:
                subblock = keras.layers.BatchNormalization()(subblock)
            subblock = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(subblock)
            subblock_input = _add_residual_shortcut(subblock_input, subblock)
        filters *= 2

    # Output convolutions
    output_tensor = subblock_input
    if batch_norm:
        output_tensor = keras.layers.BatchNormalization()(output_tensor)
    output_tensor = keras.layers.Conv2D(
        filters=num_outputs, kernel_size=(1, 1), padding='same')(output_tensor)
    return keras.models.Model(inputs=[input_tensor], outputs=[output_tensor])


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
