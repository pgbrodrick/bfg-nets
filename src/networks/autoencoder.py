import keras

from src.networks.architectures.losses import cropped_mse


# TODO:  add to config
# Model parameters
_DEFAULT_OPTIMIZER = 'adam'
_LOSS_BUFFER_CONSTANT = 0.25

# Architecture parameters
_MIN_IMAGE_DIM = 16

# Convolution parameters
_UNITS_STARTING = 64
_UNITS_MULTIPLIER = 2
_CONV_KERNEL = (3, 3)
_CONV_ACTIV = 'relu'
_CONV_PADDING = 'same'

# Pooling/upsampling parameters
_POOL_FACTOR = 2
_POOL_SIZE = (_POOL_FACTOR, _POOL_FACTOR)
_UP_SIZE = (_POOL_FACTOR, _POOL_FACTOR)

# TODO:  we'll need to incorporate this in the config so that we have flexible loss functions
# Objects for loading
custom_objects = {'_cropped_mse': cropped_mse(_LOSS_BUFFER_CONSTANT)}


def create_model(image_dim=None, num_channels=None, num_timesteps=None, output_depth=None, num_sublayers=None):
    components = _create_encoder_layers(
        image_dim=image_dim, num_channels=num_channels, num_timesteps=num_timesteps, num_sublayers=num_sublayers
    )
    components = _create_decoder_layers(output_depth=output_depth, num_sublayers=num_sublayers, **components)
    model = keras.models.Model(inputs=components['inputs'], outputs=components['decoder'])
    model.compile(optimizer=_DEFAULT_OPTIMIZER, loss=cropped_mse(_LOSS_BUFFER_CONSTANT))
    return model


def _create_encoder_layers(image_dim, num_channels, num_timesteps, num_sublayers):
    # Calculate units for each layer in encoder
    layer_units = [_UNITS_STARTING]
    width = image_dim
    while True:
        width /= _POOL_FACTOR
        if width < _MIN_IMAGE_DIM:
            break
        layer_units.append(layer_units[-1] * _UNITS_MULTIPLIER)
    # Create encoder layers
    inputs = keras.Input(shape=(image_dim, image_dim, num_channels * num_timesteps), name='input')
    layers_prepool = list()
    layers_pooled = [inputs]
    for idx_layer, units in enumerate(layer_units):
        encoder = layers_pooled[-1]
        for idx_sublayer in range(num_sublayers):
            encoder = keras.layers.Conv2D(
                units, kernel_size=_CONV_KERNEL, activation=_CONV_ACTIV, padding=_CONV_PADDING)(encoder)
            if idx_sublayer == num_sublayers - 1:
                # Pool only on last convolution of layer, store prepooled layer for decoder
                layers_prepool.append(encoder)
                encoder = keras.layers.MaxPooling2D(pool_size=_POOL_SIZE)(encoder)
            encoder = keras.layers.BatchNormalization()(encoder)
        # Store pooled layer for decoder at end of layer
        layers_pooled.append(encoder)
    return {'inputs': inputs, 'encoder': encoder, 'layer_units': layer_units, 'layers_prepool': layers_prepool}


def _create_decoder_layers(encoder, layer_units, layers_prepool, output_depth, num_sublayers):
    # Create decoder layers
    decoder = encoder
    for idx_layer, (units, pre) in enumerate(zip(reversed(layer_units), reversed(layers_prepool))):
        for idx_sublayer in range(num_sublayers):
            if idx_sublayer == num_sublayers - 1:
                # Upsample only on last convolution in sublayer
                decoder = keras.layers.UpSampling2D(size=_UP_SIZE)(decoder)
            decoder = keras.layers.Conv2D(
                units, kernel_size=_CONV_KERNEL, activation=_CONV_ACTIV, padding=_CONV_PADDING)(decoder)
            decoder = keras.layers.BatchNormalization()(decoder)
        # Combine prepooled and upsampled layers at end of layer
        decoder = keras.layers.Concatenate()([pre, decoder])
    # Create final convolution layers on top of decoder
    for idx_sublayer in range(num_sublayers):
        decoder = keras.layers.Conv2D(
            units, kernel_size=_CONV_KERNEL, activation=_CONV_ACTIV, padding=_CONV_PADDING)(decoder)
        decoder = keras.layers.BatchNormalization()(decoder)
    # Convolve to output size
    decoder = keras.layers.Conv2D(
        output_depth, kernel_size=(1, 1), activation=_CONV_ACTIV, padding=_CONV_PADDING, name='output')(decoder)
    return decoder
