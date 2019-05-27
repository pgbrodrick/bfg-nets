import rsCNN.configs.configs
import rsCNN.architectures
import rsCNN.utils
import rsCNN.utils.logging


_logger = rsCNN.utils.logging.get_root_logger()
_logger.setLevel('DEBUG')


if __name__ == '__main__':
    _logger.info('Create config templates')
    architecture_names = rsCNN.architectures.get_available_architectures()
    for architecture_name in architecture_names:
        _logger.info('Create config template for {} architecture'.format(architecture_name))
        rsCNN.configs.configs.create_config_template(
            architecture_name, rsCNN.utils.DIR_TEMPLATES, architecture_name + '.yaml')
