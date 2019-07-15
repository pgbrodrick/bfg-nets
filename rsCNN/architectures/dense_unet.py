from typing import Tuple

import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, ReLU, UpSampling2D

from rsCNN.architectures import config_sections, network_sections


# TODO:  implement optional bottleneck layers


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
        internal_activation: str = config_sections.DEFAULT_INTERNAL_ACTIVATION,
        kernel_size: Tuple[int, int] = config_sections.DEFAULT_KERNEL_SIZE,
        padding: str = config_sections.DEFAULT_PADDING,
        pool_size: Tuple[int, int] = config_sections.DEFAULT_POOL_SIZE,
        use_batch_norm: bool = config_sections.DEFAULT_USE_BATCH_NORM,
        use_growth: bool = config_sections.DEFAULT_USE_GROWTH,
        use_initial_colorspace_transformation_layer: bool =
        config_sections.DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER
) -> keras.models.Model:


    conv2d_options = {'filters': filters,
                      'kernel_size': kernel_size,
                      'padding': padding,
                      'activation': internal_activation,
                      'use_batch_norm': use_batch_norm}

    transition_options = conv2d_options.copy()
    transition_options['kernel_size'] = (1,1)

    # Initial convolution
    inlayer = keras.layers.Input(shape=inshape)
    encoder = inlayer

    # Optional colorspace transformation (not in block format)
    if use_initial_colorspace_transformation_layer:
        encoder = network_sections.colorspace_transformation(inshape, encoder, use_batch_norm)

    # Encoding, block-wise
    passthrough_layers = list()
    for num_layers in block_structure:

        # Create a dense block
        encoder = network_sections.dense_2d_block(encoder, conv2d_options, num_layers)
        passthrough_layers.append(encoder)

        # Add a transition block
        encoder = network_sections.Conv2D_Options(encoder, transition_options)

        # Pool
        encoder = MaxPooling2D(pool_size=pool_size)(encoder)

        if use_growth:
            conv2d_options['filters'] *= 2
            transition_options['filters'] *= 2

    # Encoder/Decoder Transition Block
    transition = network_sections.dense_2d_block(encoder, conv2d_options, block_structure[-1])

    decoder = transition
    # Decoding, block-wise
    for num_layers, layer_passed_through in zip(reversed(block_structure), reversed(passthrough_layers)):

        if use_growth:
            conv2d_options['filters'] = int(conv2d_options['filters'] / 2)
            transition_options['filters'] = int(transition_options['filters'] / 2)

        # Upsample
        decoder = UpSampling2D(size=pool_size, interpolation='bilinear')(decoder)

        # Create dense block and concatenate
        decoder = network_sections.Conv2D_Options(decoder, conv2d_options)
        decoder = Concatenate()([layer_passed_through, decoder])

        # Add a transition block
        decoder = network_sections.dense_2d_block(decoder, transition_options, num_layers)

    # Output convolutions
    output_layer = decoder
    output_layer = network_sections.Conv2D_Options(output_layer, conv2d_options)
    output_layer = Conv2D(filters=n_classes, kernel_size=(1, 1), padding='same', activation=output_activation)(output_layer)
    return keras.models.Model(inputs=[inlayer], outputs=[output_layer])
