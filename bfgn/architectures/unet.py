from typing import Tuple

import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

from bfgn.architectures import config_sections, network_sections


class ArchitectureConfigSection(
    config_sections.AutoencoderMixin,
    config_sections.BlockMixin,
    config_sections.GrowthMixin,
    config_sections.BaseArchitectureConfigSection,
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
    use_initial_colorspace_transformation_layer: bool = config_sections.DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER,
) -> keras.models.Model:
    """ Construct a U-net style network with flexible shape
    """

    input_width = inshape[0]

    # TODO: Move assertion to configs check
    minimum_width = input_width / 2 ** len(block_structure)
    assert minimum_width >= 2, (
        "The convolution width in the last encoding block ({}) is less than 2."
        + "Reduce the number of blocks in block_structure (currently {}).".format(
            len(block_structure)
        )
    )

    conv2d_options = {
        "filters": filters,
        "kernel_size": kernel_size,
        "padding": padding,
        "activation": internal_activation,
        "use_batch_norm": use_batch_norm,
    }

    layers_pass_through = list()

    inlayer = keras.layers.Input(shape=inshape)
    encoder = inlayer

    if use_initial_colorspace_transformation_layer:
        encoder = network_sections.colorspace_transformation(
            inshape, encoder, use_batch_norm
        )

    # Encoding Layers
    # Each encoder block has a number of subblocks
    for num_sublayers in block_structure:
        for _sublayer in range(num_sublayers):
            # Each subblock has a number of convolutions
            encoder = network_sections.Conv2D_Options(encoder, conv2d_options)
        # Each encoder block passes its pre-pooled layers through to the decoder
        layers_pass_through.append(encoder)
        encoder = MaxPooling2D(pool_size=pool_size)(encoder)
        if use_growth:
            conv2d_options["filters"] *= 2

    # Transition Layers
    transition = encoder
    for _sublayer in range(block_structure[-1]):
        transition = network_sections.Conv2D_Options(transition, conv2d_options)

    # Decoding Layers
    decoder = transition
    # Each decoder block has a number of subblocks, but in reverse order of encoder
    for num_subblocks, layer_passed_through in zip(
        reversed(block_structure), reversed(layers_pass_through)
    ):
        if use_growth:
            conv2d_options["filters"] = int(conv2d_options["filters"] / 2)

        decoder = UpSampling2D(size=pool_size, interpolation="bilinear")(decoder)
        decoder = network_sections.Conv2D_Options(decoder, conv2d_options)
        decoder = Concatenate()([layer_passed_through, decoder])

        for _sublayer in range(num_subblocks):
            # Each subblock has a number of convolutions
            decoder = network_sections.Conv2D_Options(decoder, conv2d_options)

    # Output convolutions
    output_layer = decoder
    output_layer = network_sections.Conv2D_Options(output_layer, conv2d_options)
    output_layer = Conv2D(
        filters=n_classes,
        kernel_size=(1, 1),
        padding="same",
        activation=output_activation,
    )(output_layer)
    return keras.models.Model(inputs=[inlayer], outputs=[output_layer])
