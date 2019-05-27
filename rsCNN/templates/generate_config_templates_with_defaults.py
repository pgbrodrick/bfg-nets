import os

from rsCNN import architectures, utils
from rsCNN.configuration import configs
from rsCNN.utils import logging


_logger = utils.logging.get_root_logger()
_logger.setLevel('DEBUG')


if __name__ == '__main__':
    _logger.info('Create config templates')
    architecture_names = architectures.get_available_architectures()
    for architecture_name in architecture_names:
        _logger.info('Create config template for {} architecture'.format(architecture_name))
        configs.create_config_template(
            architecture_name, os.path.join(utils.DIR_TEMPLATES, architecture_name + '.yaml'))
