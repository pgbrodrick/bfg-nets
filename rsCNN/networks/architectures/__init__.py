from typing import List

from rsCNN.networks.architectures \
    import alex_net, change_detection, dense_flat_net, dense_unet, dilation_net, flat_net, residual_dilation_net, \
    residual_flat_net, residual_unet, shared, unet


_ARCHITECTURE_NAMES = [
    'alex_net', 'dense_flat_net', 'dense_unet', 'dilation_net', 'flat_net', 'residual_dilation_net',
    'residual_flat_net', 'residual_unet', 'unet'
]


def create_model_from_architecture_options(
        architecture_name: str, architecture_options: shared.BaseArchitectureOptions
) -> 'keras.models.Model':
    _assert_architecture_exists(architecture_name)
    architecture_module = globals()[architecture_name]
    kwargs = {key: getattr(architecture_options, key) for key in architecture_options.get_option_keys()}
    return architecture_module.create_model(**kwargs)


def get_architecture_options(architecture_name: str) -> shared.BaseArchitectureOptions:
    _assert_architecture_exists(architecture_name)
    architecture_module = globals()[architecture_name]
    return architecture_module.ArchitectureOptions()


def get_available_architectures() -> List[str]:
    return _ARCHITECTURE_NAMES


def _assert_architecture_exists(architecture_name: str) -> None:
    assert architecture_name in _ARCHITECTURE_NAMES, \
        'architecture does not exist ({}) in available architectures: {}'.format(
            architecture_name, ', '.join(_ARCHITECTURE_NAMES))
