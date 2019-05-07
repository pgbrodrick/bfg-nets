from typing import Tuple

import keras
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D

from rsCNN.networks.architectures import shared


DEFAULT_BLOCK_STRUCTURE = (1, 1, 1, 1)
DEFAULT_MIN_CONV_WIDTH = 8
DEFAULT_POOL_SIZE = (2, 2)
DEFAULT_USE_GROWTH = False


class ArchitectureOptions(shared.BaseArchitectureOptions):
    dilation_rate = None
    num_layers = None

    def __init__(self):
        self._field_defaults.extend([
            ('block_structure', DEFAULT_BLOCK_STRUCTURE, tuple),
            ('min_conv_width', DEFAULT_MIN_CONV_WIDTH, int),
            ('pool_size', DEFAULT_POOL_SIZE, tuple),
            ('use_growth', DEFAULT_USE_GROWTH, bool),
        ])
        super().__init__()


def create_model(
        inshape: Tuple[int, int, int],
        n_classes: int,
        output_activation: str,
        block_structure: Tuple[int, ...] = DEFAULT_BLOCK_STRUCTURE,
        filters: int = shared.DEFAULT_FILTERS,
        kernel_size: Tuple[int, int] = shared.DEFAULT_KERNEL_SIZE,
        min_conv_width: int = DEFAULT_MIN_CONV_WIDTH,
        padding: str = shared.DEFAULT_PADDING,
        pool_size: Tuple[int, int] = DEFAULT_POOL_SIZE,
        use_batch_norm: bool = shared.DEFAULT_USE_BATCH_NORM,
        use_growth: bool = DEFAULT_USE_GROWTH,
        use_initial_colorspace_transformation_layer: bool = shared.DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER
) -> keras.models.Model:

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
    for num_subblocks in block_structure:
        for idx_sublayer in range(num_subblocks):
            # Each subblock has two convolutions
            encoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(encoder)
            if use_batch_norm:
                encoder = BatchNormalization()(encoder)
            encoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(encoder)
            if use_batch_norm:
                encoder = BatchNormalization()(encoder)
        # Each encoder block passes its pre-pooled layers through to the decoder
        layers_pass_through.append(encoder)
        encoder = MaxPooling2D(pool_size=pool_size)(encoder)
        if use_growth:
            filters *= 2

    encoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(encoder)
    output_layer = Flatten()(encoder)
    output_layer = Dense(units=filters)(output_layer)
    output_layer = Dense(units=filters)(output_layer)
    output_layer = Dense(units=n_classes, activation=output_activation)(output_layer)
    return keras.models.Model(inputs=[inlayer], outputs=[output_layer])
