from typing import Tuple

import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, MaxPooling2D, UpSampling2D


DEFAULT_BLOCK_STRUCTURE = (2, 2, 2, 2)
DEFAULT_BATCH_NORM = True
DEFAULT_INITIAL_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_MIN_CONV_WIDTH = 8
DEFAULT_PADDING = 'same'
DEFAULT_POOL_SIZE = (2, 2)
DEFAULT_STRIDES = (1, 1)


def parse_architecture_options(**kwargs):
    return {
        'block_structure': kwargs.get('block_structure', DEFAULT_BLOCK_STRUCTURE),
        'batch_norm': kwargs.get('batch_norm', DEFAULT_BATCH_NORM),
        'initial_filters': kwargs.get('initial_filters', DEFAULT_INITIAL_FILTERS),
        'kernel_size': kwargs.get('kernel_size', DEFAULT_KERNEL_SIZE),
        'min_conv_width': kwargs.get('min_conv_width', DEFAULT_MIN_CONV_WIDTH),
        'padding': kwargs.get('padding', DEFAULT_PADDING),
        'pool_size': kwargs.get('pool_size', DEFAULT_POOL_SIZE),
    }


def create_model(
        input_shape: Tuple[int, int, int],
        num_outputs: int,
        output_activation: str,
        block_structure: Tuple[int, ...] = DEFAULT_BLOCK_STRUCTURE,
        batch_norm: bool = DEFAULT_BATCH_NORM,
        initial_filters: int = DEFAULT_INITIAL_FILTERS,
        kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
        min_conv_width: int = DEFAULT_MIN_CONV_WIDTH,
        padding: str = DEFAULT_PADDING,
        pool_size: Tuple[int, int] = DEFAULT_POOL_SIZE,
        use_growth: bool = False,
):
    """ Construct a U-net style network with flexible shape

    Arguments:
    input_shape - tuple/list
      Designates the input shape of an image to be passed to
      the network.
    n_classes - int
      The number of classes the network is meant to classify.
    filters - int/str
      If integer, a fixed number of convolution filters to use
      in the network.  If 'growth' tells the network to grow
      in depth to maintain a constant number of neurons.
    batch_norm - bool
      Whether or not to use batch normalization after each layer.

    Returns:
      A U-net style network keras network.
    """
    input_width = input_shape[0]
    minimum_width = input_width / 2 ** len(block_structure)
    assert minimum_width > min_conv_width, \
        'The convolution width in the last encoding block ({}) is less than '.format(minimum_width) + \
        'the minimum specified width ({}). Either reduce '.format(min_conv_width) + \
        'the number of blocks in block_structure (currently {}) or '.format(len(block_structure)) + \
        'the value of min_conv_width.'

    # Need to track the following throughout the model creation
    filters = initial_filters
    layers_pass_through = list()

    # Encodings
    input_layer = keras.layers.Input(shape=input_shape)
    encoder = input_layer
    # Each encoder block has a number of subblocks
    for num_subblocks in block_structure:
        for idx_sublayer in range(num_subblocks):
            # Each subblock has two convolutions
            encoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(encoder)
            if (batch_norm):
                encoder = BatchNormalization()(encoder)
            encoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(encoder)
            if (batch_norm):
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
            # Each subblock has two convolutions
            decoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(decoder)
            if (batch_norm):
                decoder = BatchNormalization()(decoder)
            decoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(decoder)
            if (batch_norm):
                decoder = BatchNormalization()(decoder)
        decoder = UpSampling2D(size=pool_size)(decoder)
        decoder = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(decoder)
        if (batch_norm):
            decoder = BatchNormalization()(decoder)
        decoder = Concatenate()([layer_passed_through, decoder])
        if use_growth:
            filters /= 2

    # Last convolutions
    output_layer = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(decoder)
    if (batch_norm):
        output_layer = BatchNormalization()(output_layer)
    output_layer = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(output_layer)
    if (batch_norm):
        output_layer = BatchNormalization()(output_layer)
    output_layer = Conv2D(
        filters=num_outputs, kernel_size=(1, 1), padding='same', activation=output_activation)(output_layer)
    return keras.models.Model(inputs=[input_layer], outputs=[output_layer])

