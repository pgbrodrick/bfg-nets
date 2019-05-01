import keras

from rsCNN.networks.architectures \
    import change_detection, dilation_net, flat_net, residual_dilation_net, residual_flat_net, residual_unet, unet
from rsCNN.networks.architectures import shared


_architecture_names = (
    'dilation_net', 'flat_net', 'residual_dilation_net', 'residual_flat_net', 'residual_unet', 'unet'
)


def create_model_from_options(
        architecture_name: str, architecture_options: shared.BaseArchitectureOptions
) -> keras.models.Model:
    assert check_architecture_exists(architecture_name), \
        'Architecture does not exist ({}) in options: {}'.format(architecture_name, ', '.join(_architecture_names))
    kwargs = {field_name: getattr(architecture_options, field_name)
              for field_name, _, _ in architecture_options._field_defaults}
    return globals()[architecture_name].create_model(**kwargs)


def check_architecture_exists(architecture_name: str) -> bool:
    return architecture_name in _architecture_names
