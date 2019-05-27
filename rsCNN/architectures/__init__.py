from typing import List


_ARCHITECTURE_NAMES = [
    'alex_net', 'dense_flat_net', 'dense_unet', 'dilation_net', 'flat_net', 'residual_dilation_net',
    'residual_flat_net', 'residual_unet', 'unet'
]


def get_available_architectures() -> List[str]:
    """Gets a list of available architectures.

    Returns:
        List of available architectures
    """
    return _ARCHITECTURE_NAMES
