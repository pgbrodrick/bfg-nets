import configparser
import os

from rsCNN.utils import DIR_TEMPLATES
from rsCNN.networks.architectures import regress_net, residual_net, residual_unet, unet


def create_config_file_from_architecture_defaults():
    for arch_module in (regress_net, residual_net, residual_unet, unet):
        module_name = arch_module.__name__.split('.')[-1]
        config = configparser.ConfigParser()
        config[module_name] = arch_module.parse_architecture_options(**dict())
        with open(os.path.join(DIR_TEMPLATES, module_name + '.ini'), 'w') as file_:
            config.write(file_)
