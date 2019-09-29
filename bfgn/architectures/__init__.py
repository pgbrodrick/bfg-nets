import os
from typing import List


def get_available_architectures() -> List[str]:
    """Gets a list of available architectures.

    Returns:
        List of available architectures.
    """
    dir_arch = os.path.dirname(__file__)
    return sorted([os.path.splitext(path)[0] for path in os.listdir(dir_arch) if path.endswith("net.py")])
