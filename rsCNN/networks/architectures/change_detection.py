from typing import Tuple

import keras
from keras.layers import BatchNormalization, Concatenate, Conv2D, MaxPooling2D, UpSampling2D


DEFAULT_BLOCK_STRUCTURE = (1, 1, 1, 1)
DEFAULT_USE_BATCH_NORM = True
DEFAULT_INITIAL_FILTERS = 64
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_MIN_CONV_WIDTH = 8
DEFAULT_PADDING = 'same'
DEFAULT_POOL_SIZE = (2, 2)
DEFAULT_USE_GROWTH = False


# TODO:  this script should be generalized and not called something as specific as "change detection". Also, I need to
#  test whether it makes more sense to just remove this and instead use a u-net with four bands instead of a this u-net
#  with two images with two band each.


def parse_architecture_options(**kwargs):
    return {
        'block_structure': kwargs.get('block_structure', DEFAULT_BLOCK_STRUCTURE),
        'initial_filters': kwargs.get('initial_filters', DEFAULT_INITIAL_FILTERS),
        'kernel_size': kwargs.get('kernel_size', DEFAULT_KERNEL_SIZE),
        'min_conv_width': kwargs.get('min_conv_width', DEFAULT_MIN_CONV_WIDTH),
        'padding': kwargs.get('padding', DEFAULT_PADDING),
        'pool_size': kwargs.get('pool_size', DEFAULT_POOL_SIZE),
        'use_batch_norm': kwargs.get('use_batch_norm', DEFAULT_USE_BATCH_NORM),
        'use_growth': kwargs.get('use_growth', DEFAULT_USE_GROWTH),
    }


def create_model(
        inshape: Tuple[int, int, int],
        n_classes: int,
        output_activation: str,
        block_structure: Tuple[int, ...] = DEFAULT_BLOCK_STRUCTURE,
        initial_filters: int = DEFAULT_INITIAL_FILTERS,
        kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
        min_conv_width: int = DEFAULT_MIN_CONV_WIDTH,
        padding: str = DEFAULT_PADDING,
        pool_size: Tuple[int, int] = DEFAULT_POOL_SIZE,
        use_batch_norm: bool = DEFAULT_USE_BATCH_NORM,
        use_growth: bool = False
) -> keras.models.Model:
    model = ImageChangeDetection(
        inshape=inshape,
        n_classes=n_classes,
        output_activation=output_activation,
        block_structure=block_structure,
        initial_filters=initial_filters,
        kernel_size=kernel_size,
        min_conv_width=min_conv_width,
        padding=padding,
        pool_size=pool_size,
        use_batch_norm=use_batch_norm,
        use_growth=use_growth
    )
    return model.model


class ImageChangeDetection(object):
    """
    Detect changes between image pairs with supervised training. Learn weights to generate features from the images,
    then use those features to determine whether the images represent change or no change.
    """

    def __init__(
            self,
            inshape: Tuple[int, int, int],
            n_classes: int,
            output_activation: str,
            block_structure: Tuple[int, ...] = DEFAULT_BLOCK_STRUCTURE,
            initial_filters: int = DEFAULT_INITIAL_FILTERS,
            kernel_size: Tuple[int, int] = DEFAULT_KERNEL_SIZE,
            min_conv_width: int = DEFAULT_MIN_CONV_WIDTH,
            padding: str = DEFAULT_PADDING,
            pool_size: Tuple[int, int] = DEFAULT_POOL_SIZE,
            use_batch_norm: bool = DEFAULT_USE_BATCH_NORM,
            use_growth: bool = False
    ):
        # Checks
        input_width = inshape[0]
        minimum_width = input_width / 2 ** len(block_structure)
        assert minimum_width >= min_conv_width, \
            'The convolution width in the last encoding block ({}) is less than '.format(minimum_width) + \
            'the minimum specified width ({}). Either reduce '.format(min_conv_width) + \
            'the number of blocks in block_structure (currently {}) or '.format(len(block_structure)) + \
            'the value of min_conv_width.'
        # Parameters
        self.inshape = inshape
        self.n_classes = n_classes
        self.output_activation = output_activation
        self.block_structure = block_structure
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size
        self.min_conv_width = min_conv_width
        self.padding = padding
        self.pool_size = pool_size
        self.use_batch_norm = use_batch_norm
        self.use_growth = use_growth
        # State to track through model
        self._layers_pass_through = list()
        self._current_filters = initial_filters
        # Create model
        self.model = self._create_model()

    def _create_model(self):
        # Input layers
        image_t0 = keras.layers.Input(shape=self.inshape)
        image_t1 = keras.layers.Input(shape=self.inshape)
        # Network
        shared_encoder_layers = self._create_shared_encoder_layers()
        encoded_t0, pass_through_t0 = self._apply_encoder_to_input(image_t0, shared_encoder_layers)
        encoded_t1, pass_through_t1 = self._apply_encoder_to_input(image_t1, shared_encoder_layers)
        output = self._decode_and_contrast_encodings(encoded_t0, pass_through_t0, encoded_t1, pass_through_t1)
        # Model
        return keras.models.Model(inputs=[image_t0, image_t1], outputs=[output])

    def _create_shared_encoder_layers(self):
        shared_layers = list()
        # Each encoder block has a number of subblocks
        for idx_encoder_block, num_subblocks in enumerate(self.block_structure):
            for idx_sublayer in range(num_subblocks):
                # Each subblock has two convolutions
                shared_layers.append(
                    Conv2D(filters=self._current_filters, kernel_size=self.kernel_size, padding=self.padding)
                )
                if self.use_batch_norm:
                    shared_layers.append(BatchNormalization())
                shared_layers.append(
                    Conv2D(filters=self._current_filters, kernel_size=self.kernel_size, padding=self.padding)
                )
                if self.use_batch_norm:
                    shared_layers.append(BatchNormalization())
            # Each encoder block passes its pre-pooled layers through to the decoder
            shared_layers[-1].name = 'encoder_{}'.format(idx_encoder_block)
            shared_layers.append(MaxPooling2D(pool_size=self.pool_size))
            if self.use_growth:
                self._current_filters *= 2
        return shared_layers

    def _apply_encoder_to_input(self, input_image, shared_encoder_layers):
        encoded = input_image
        layers_pass_through = list()
        for layer in shared_encoder_layers:
            encoded = layer(encoded)
            if layer.name.startswith('encoder'):
                layers_pass_through.append(encoded)
        return encoded, layers_pass_through

    def _decode_and_contrast_encodings(self, encoded_t0, pass_through_t0, encoded_t1, pass_through_t1):
        decoder = keras.layers.Concatenate()([encoded_t0, encoded_t1])
        # Each decoder block has a number of subblocks, but in reverse order of encoder
        for idx_decoder_block, (num_subblocks, pt_t0, pt_t1) \
                in enumerate(zip(reversed(self.block_structure), reversed(pass_through_t0), reversed(pass_through_t1))):
            for idx_sublayer in range(num_subblocks):
                # Each subblock has two convolutions
                decoder = Conv2D(
                    filters=self._current_filters, kernel_size=self.kernel_size, padding=self.padding)(decoder)
                if self.use_batch_norm:
                    decoder = BatchNormalization()(decoder)
                decoder = Conv2D(
                    filters=self._current_filters, kernel_size=self.kernel_size, padding=self.padding)(decoder)
                if self.use_batch_norm:
                    decoder = BatchNormalization()(decoder)
            decoder = UpSampling2D(size=self.pool_size)(decoder)
            decoder = Conv2D(filters=self._current_filters, kernel_size=self.kernel_size, padding=self.padding)(decoder)
            if self.use_batch_norm:
                decoder = BatchNormalization()(decoder)
            decoder = Concatenate()([decoder, pt_t0, pt_t1])
            if self.use_growth:
                self._current_filters = int(self._current_filters / 2)
        # Last convolutions
        classifier = Conv2D(
            filters=self._current_filters, kernel_size=self.kernel_size, padding=self.padding)(decoder)
        if self.use_batch_norm:
            classifier = BatchNormalization()(classifier)
        classifier = Conv2D(
            filters=self._current_filters, kernel_size=self.kernel_size, padding=self.padding)(classifier)
        if self.use_batch_norm:
            classifier = BatchNormalization()(classifier)
        return Conv2D(self.n_classes, kernel_size=(1, 1), padding='same', activation=self.output_activation)(classifier)
