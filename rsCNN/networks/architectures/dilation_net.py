from typing import List, Tuple, Union

import keras

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization


DEFAULT_DILATION_RATE = 2
DEFAULT_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_NUM_LAYERS = 8
DEFAULT_PADDING = 'same'
DEFAULT_USE_BATCH_NORM = True
DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER = False


def parse_architecture_options(**kwargs):
    return {
        'dilation_rate': kwargs.get('dilation_rate', DEFAULT_DILATION_RATE),
        'filters': kwargs.get('filters', DEFAULT_FILTERS),
        'kernel_size': kwargs.get('kernel_size', DEFAULT_KERNEL_SIZE),
        'num_layers': kwargs.get('num_layers', DEFAULT_NUM_LAYERS),
        'padding': kwargs.get('padding', DEFAULT_PADDING),
        'use_batch_norm': kwargs.get('use_batch_norm', DEFAULT_USE_BATCH_NORM),
        'use_initial_colorspace_transformation_layer':
            kwargs.get('use_initial_colorspace_transformation_layer',
                       DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER)
    }


def create_model(
        inshape: Tuple[int, int, int],
        n_classes: int,
        output_activation: str,
        dilation_rate: int = DEFAULT_DILATION_RATE,
        filters: int = DEFAULT_FILTERS,
        kernel_size: Union[Tuple[int, int], List[Tuple[int, int]]] = DEFAULT_KERNEL_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        padding: str = DEFAULT_PADDING,
        use_batch_norm: bool = DEFAULT_USE_BATCH_NORM,
        use_initial_colorspace_transformation_layer: bool = DEFAULT_USE_INITIAL_COLORSPACE_TRANSFORMATION_LAYER
) -> keras.models.Model:
    inlayer = keras.layers.Input(inshape)

    if type(kernel_size) is tuple:
        kernel_sizes = [kernel_size] * num_layers
    else:
        assert len(kernel_size) == num_layers, 'If providing a list of kernel sizes, length must equal num_layers'
        kernel_sizes = kernel_size

    conv = inlayer
    if use_initial_colorspace_transformation_layer:
        intermediate_color_depth = int(inshape[-1] ** 2)
        conv = Conv2D(filters=intermediate_color_depth, kernel_size=(1, 1), padding='same')(inlayer)
        conv = Conv2D(filters=inshape[-1], kernel_size=(1, 1), padding='same')(conv)
        conv = BatchNormalization()(conv)

    for kernel_size in kernel_sizes:
        conv = Conv2D(filters=filters, dilation_rate=dilation_rate, kernel_size=kernel_size, padding=padding)(conv)
        conv = Conv2D(filters=filters, dilation_rate=dilation_rate, kernel_size=kernel_size, padding=padding)(conv)
        if use_batch_norm:
            conv = BatchNormalization()(conv)

    output_layer = Conv2D(n_classes, (1, 1), activation=output_activation, padding=padding)(conv)
    model = keras.models.Model(inputs=inlayer, outputs=output_layer)
    return model
