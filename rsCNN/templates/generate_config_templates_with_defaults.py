from rsCNN import configs
from rsCNN.networks import architectures
from rsCNN.utils import DIR_TEMPLATES, logging


_logger = logging.get_root_logger()
_logger.setLevel('DEBUG')


if __name__ == '__main__':
    _logger.info('Create config templates')
    architecture_names = architectures.get_available_architectures()
    for architecture_name in architecture_names:
        _logger.info('Create config template for {} architecture'.format(architecture_name))
        configs.create_config_template(architecture_name, DIR_TEMPLATES, architecture_name + '.yaml')
