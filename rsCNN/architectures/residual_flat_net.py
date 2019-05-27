from typing import Tuple

import keras
from keras.layers import BatchNormalization, Conv2D

import rsCNN.architectures.config_sections


class ArchitectureConfigSection(
    rsCNN.architectures.config_sections.BlockMixin,
    rsCNN.architectures.config_sections.BaseArchitectureConfigSection
):
    pass


def create_model(
        inshape: Tuple[int, int, int],
        n_classes: int,
        output_activation: str,
        block_structure: Tuple[int, ...] = rsCNN.architectures.config_sections.DEFAULT_BLOCK_STRUCTURE,
        filters: int = rsCNN.architectures.config_sections.DEFAULT_FILTERS,
        kernel_size: Tuple[int, int] = rsCNN.architectures.config_sections.DEFAULT_KERNEL_SIZE,
        padding: str = rsCNN.architectures.config_sections.DEFAULT_PADDING,
        use_batch_norm: bool = rsCNN.architectures.config_sections.DEFAULT_USE_BATCH_NORM,
        use_initial_colorspace_transformation_layer: bool =
    rsCNN.architectures.config_sections.DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER
) -> keras.models.Model:

    # Initial convolution
    inlayer = keras.layers.Input(shape=inshape)

    if use_initial_colorspace_transformation_layer:
        intermediate_color_depth = int(inshape[-1] ** 2)
        conv = Conv2D(filters=intermediate_color_depth, kernel_size=(1, 1), padding='same')(inlayer)
        conv = Conv2D(filters=inshape[-1], kernel_size=(1, 1), padding='same')(conv)
        conv = BatchNormalization()(conv)
    else:
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inlayer)

    # Iterate blocks and subblocks
    subblock_input = conv
    for idx_block, num_subblocks in enumerate(block_structure):
        subblock = subblock_input
        for idx_sublayer in range(num_subblocks):
            subblock = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(subblock)
            if use_batch_norm:
                subblock = BatchNormalization()(subblock)
        subblock_input = _add_residual_shortcut(subblock_input, subblock)
        filters *= 2

    # Output convolutions
    output_layer = Conv2D(
        filters=n_classes, kernel_size=(1, 1), padding='same', activation=output_activation)(subblock_input)
    return keras.models.Model(inputs=[inlayer], outputs=[output_layer])


def _add_residual_shortcut(input_layer: keras.layers.Layer, residual_module: keras.layers.Layer):
    """
    Adds a shortcut connection by combining a input tensor and residual module
    """
    shortcut = input_layer

    # We need to apply a convolution if the input and block shapes do not match, every block transition
    inshape = keras.backend.int_shape(input_layer)[1:]
    residual_shape = keras.backend.int_shape(residual_module)[1:]
    if inshape != residual_shape:
        strides = (int(round(inshape[0] / residual_shape[0])), int(round(inshape[1] / residual_shape[1])))
        shortcut = Conv2D(
            filters=residual_shape[-1], kernel_size=(1, 1), padding='valid', strides=strides)(shortcut)
    return keras.layers.add([shortcut, residual_module])
