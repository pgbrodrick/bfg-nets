import os

from bfgn import architectures, utils
from bfgn.configuration import configs


def test_create_config_templates_in_hacky_manner_so_sorry() -> None:
    architecture_names = architectures.get_available_architectures()
    for architecture_name in architecture_names:
        configs.create_config_template(
            architecture_name, os.path.join(utils.DIR_TEMPLATES, architecture_name + ".yaml")
        )
