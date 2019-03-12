from typing import Iterable

import keras


def create_residual_network(
        input_shape: Iterable[int],
        num_outputs: int,
        block_structure: Iterable[int] = (2, 2, 2, 2),
        batch_norm: bool = True,
        initial_filters: int = 64,
        kernel_size: Iterable[int] = (3, 3),
        padding: str = 'same',
        pool_size: Iterable[int] = (3, 3),
        strides: Iterable[int] = (1, 1),
)-> keras.models.Model:

    # Initial convolution
    input_tensor = keras.layers.Input(shape=input_shape)
    conv = keras.layers.Conv2D(
        filters=initial_filters, kernel_size=kernel_size, padding=padding, strides=strides)(input_tensor)
    if batch_norm:
        conv = keras.layers.BatchNormalization()(conv)
    pool = keras.layers.MaxPooling2D(pool_size)(conv)

    # Iterate blocks and subblocks
    subblock_input = pool
    filters = initial_filters
    for idx_block, num_subblocks in enumerate(block_structure):
        for idx_sublayer in range(num_subblocks):
            is_first_subblock_in_first_block = idx_block == 0 and idx_sublayer == 0
            subblock = subblock_input
            if batch_norm and not is_first_subblock_in_first_block:
                subblock = keras.layers.BatchNormalization()(subblock)
            subblock = keras.layers.Conv2D(
                filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(subblock)
            subblock_input = _add_residual_shortcut(subblock_input, subblock)
        filters *= 2

    # Output convolutions
    output_tensor = subblock_input
    if batch_norm:
        output_tensor = keras.layers.BatchNormalization()(output_tensor)
    output_tensor = keras.layers.Conv2D(
        filters=num_outputs, kernel_size=(1, 1), padding='same')(output_tensor)
    return keras.models.Model(inputs=[input_tensor], outputs=[output_tensor])


def _add_residual_shortcut(input_tensor, residual_module):
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
