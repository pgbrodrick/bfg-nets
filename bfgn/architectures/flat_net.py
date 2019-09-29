from typing import Tuple

import keras
from keras.layers import BatchNormalization, Conv2D

from bfgn.architectures import config_sections


class ArchitectureConfigSection(
    config_sections.FlatMixin, config_sections.BaseArchitectureConfigSection
):
    pass


def create_model(
    inshape: Tuple[int, int, int],
    n_classes: int,
    output_activation: str,
    filters: int = config_sections.DEFAULT_FILTERS,
    internal_activation: str = config_sections.DEFAULT_INTERNAL_ACTIVATION,
    kernel_size: Tuple[int, int] = config_sections.DEFAULT_KERNEL_SIZE,
    num_layers: int = config_sections.DEFAULT_NUM_LAYERS,
    padding: str = config_sections.DEFAULT_PADDING,
    use_batch_norm: bool = config_sections.DEFAULT_USE_BATCH_NORM,
    use_initial_colorspace_transformation_layer: bool = config_sections.DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER,
) -> keras.models.Model:
    """ Construct a flat style network with flexible shape
    """
    inlayer = keras.layers.Input(inshape)

    conv = inlayer
    if use_initial_colorspace_transformation_layer:
        intermediate_color_depth = int(inshape[-1] ** 2)
        conv = Conv2D(
            filters=intermediate_color_depth, kernel_size=(1, 1), padding="same"
        )(inlayer)
        conv = Conv2D(filters=inshape[-1], kernel_size=(1, 1), padding="same")(conv)
        conv = BatchNormalization()(conv)

    for idx_layer in range(num_layers):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(conv)
        if use_batch_norm:
            conv = BatchNormalization()(conv)

    output_layer = Conv2D(
        n_classes, (1, 1), activation=output_activation, padding=padding
    )(conv)
    model = keras.models.Model(inputs=inlayer, outputs=output_layer)
    return model
