from typing import Tuple

import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, MaxPooling2D, UpSampling2D

from rsCNN.architectures import config_sections


class ArchitectureConfigSection(
    config_sections.AutoencoderMixin,
    config_sections.BlockMixin,
    config_sections.GrowthMixin,
    config_sections.BaseArchitectureConfigSection
):
    pass


def create_model(
        inshape: Tuple[int, int, int],
        n_classes: int,
        output_activation: str,
        block_structure: Tuple[int, ...] = config_sections.DEFAULT_BLOCK_STRUCTURE,
        filters: int = config_sections.DEFAULT_FILTERS,
        kernel_size: Tuple[int, int] = config_sections.DEFAULT_KERNEL_SIZE,
        min_conv_width: int = config_sections.DEFAULT_MIN_CONV_WIDTH,
        padding: str = config_sections.DEFAULT_PADDING,
        pool_size: Tuple[int, int] = config_sections.DEFAULT_POOL_SIZE,
        use_batch_norm: bool = config_sections.DEFAULT_USE_BATCH_NORM,
        use_growth: bool = config_sections.DEFAULT_USE_GROWTH,
        use_initial_colorspace_transformation_layer: bool =
    config_sections.DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER
) -> keras.models.Model:
    """ Construct a U-net style network with flexible shape
    """

    input_width = inshape[0]
    minimum_width = input_width / 2 ** len(block_structure)
    assert minimum_width >= min_conv_width, \
        'The convolution width in the last encoding block ({}) is less than '.format(minimum_width) + \
        'the minimum specified width ({}). Either reduce '.format(min_conv_width) + \
        'the number of blocks in block_structure (currently {}) or '.format(len(block_structure)) + \
        'the value of min_conv_width.'

    layers_pass_through = list()

    # Encodings
    inlayer = keras.layers.Input(shape=inshape)
    encoder = inlayer

    if use_initial_colorspace_transformation_layer:
        intermediate_color_depth = int(inshape[-1] ** 2)
        encoder = Conv2D(filters=intermediate_color_depth, kernel_size=(1, 1), padding='same')(inlayer)
        encoder = Conv2D(filters=inshape[-1], kernel_size=(1, 1), padding='same')(encoder)
        encoder = BatchNormalization()(encoder)

    # Each encoder block has a number of subblocks
    for num_sublayers in block_structure:
        for idx_sublayer in range(num_sublayers):
            # Each subblock has a number of convolutions
            encoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(encoder)
            if use_batch_norm:
                encoder = BatchNormalization()(encoder)
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
            # Each subblock has a number of convolutions
            decoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(decoder)
            if use_batch_norm:
                decoder = BatchNormalization()(decoder)
        decoder = UpSampling2D(size=pool_size)(decoder)
        decoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(decoder)
        if use_batch_norm:
            decoder = BatchNormalization()(decoder)
        decoder = Concatenate()([layer_passed_through, decoder])
        if use_growth:
            filters = int(filters / 2)

    # Last convolutions
    output_layer = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(decoder)
    if use_batch_norm:
        output_layer = BatchNormalization()(output_layer)
    output_layer = Conv2D(
        filters=n_classes, kernel_size=(1, 1), padding='same', activation=output_activation)(output_layer)
    return keras.models.Model(inputs=[inlayer], outputs=[output_layer])
