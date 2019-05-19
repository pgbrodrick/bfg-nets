from typing import Tuple

import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, ReLU, UpSampling2D

from rsCNN.configs.shared import ConfigOption
from rsCNN.networks.architectures import shared


# TODO:  implement optional bottleneck layers


DEFAULT_BLOCK_STRUCTURE = (2, 2, 2, 2)
DEFAULT_FILTERS = 12
DEFAULT_POOL_SIZE = (2, 2)
DEFAULT_USE_GROWTH = False


class ArchitectureOptions(shared.BaseArchitectureOptions):
    block_structure = None
    filters = None
    pool_size = None
    use_growth = None
    _config_options_extra = [
        ConfigOption('block_structure', DEFAULT_BLOCK_STRUCTURE, tuple),
        ConfigOption('filters', DEFAULT_FILTERS, int),
        ConfigOption('pool_size', DEFAULT_POOL_SIZE, tuple),
        ConfigOption('use_growth', DEFAULT_USE_GROWTH, bool),
    ]

    def check_config_validity(self):
        # TODO:  assert that block_structure has more than one layer in each block for dense purposes, this will cause
        #  an error as written but, more importantly, it doesn't make any sense for this architecture
        return super().check_config_validity()


def create_model(
        inshape: Tuple[int, int, int],
        n_classes: int,
        output_activation: str,
        block_structure: Tuple[int, ...] = DEFAULT_BLOCK_STRUCTURE,
        filters: int = DEFAULT_FILTERS,
        kernel_size: Tuple[int, int] = shared.DEFAULT_KERNEL_SIZE,
        padding: str = shared.DEFAULT_PADDING,
        pool_size: Tuple[int, int] = DEFAULT_POOL_SIZE,
        use_batch_norm: bool = shared.DEFAULT_USE_BATCH_NORM,
        use_growth: bool = DEFAULT_USE_GROWTH,
        use_initial_colorspace_transformation_layer: bool = shared.DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER
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

    # Track linear filter increase if use_growth
    if use_growth:
        filter_increase = filters

    # Iterate through dense blocks in downsampling path
    passthrough_layers = list()
    for idx_block, num_layers in enumerate(block_structure):
        # Iterate through layers in the dense block
        input_layer = conv
        output_layers = list()
        for idx_layer in range(num_layers):
            if use_batch_norm:
                conv = BatchNormalization()(conv)
            conv = ReLU()(conv)
            if use_growth:
                # Increase number of filters for new layer
                filters += filter_increase
            conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(conv)
            output_layers.append(conv)
            # The next layer input is the concatenation of the dense block input and all layer outputs, regardless of
            # whether the next layer is in this block or the next
            conv = Concatenate()([input_layer] + output_layers)
        is_connecting_block = idx_block == len(block_structure) - 1
        if not is_connecting_block:
            passthrough_layers.append(conv)
            # Create transition layer between dense blocks, preserving filter number
            if use_batch_norm:
                conv = BatchNormalization()(conv)
            conv = Conv2D(filters=filters, kernel_size=(1, 1), padding=padding)(conv)
            conv = MaxPooling2D(pool_size=pool_size, padding=padding)(conv)

    # Iterate through dense blocks in upsampling path, not repeating the first block, which connects the downsampling
    # and upsampling paths
    for idx_block, (num_layers, passthrough) in enumerate(zip(block_structure[1:], reversed(passthrough_layers))):
        conv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding, strides=pool_size)(conv)
        conv = Concatenate()([passthrough, conv])
        # Iterate through layers in dense block
        for idx_layer in range(num_layers):
            if use_batch_norm:
                conv = BatchNormalization()(conv)
            conv = ReLU()(conv)
            # TODO:  verify this is reasonable, paper is not explicit about how they downsample feature maps
            if use_growth:
                filters -= filter_increase
            conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(conv)

    output_layer = Conv2D(n_classes, (1, 1), activation=output_activation, padding=padding)(conv)
    model = keras.models.Model(inputs=inlayer, outputs=output_layer)
    return model
