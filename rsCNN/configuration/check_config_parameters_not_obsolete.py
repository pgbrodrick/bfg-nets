import os
import shlex
import subprocess
from typing import List

from rsCNN import architectures, utils
from rsCNN.configuration import configs, sections
from rsCNN.utils import logging


_logger = utils.logging.get_root_logger()
_logger.setLevel('INFO')


def check_config_parameters_not_obsolete():
    _logger.info('Check config parameters are not obsolete')
    _logger.warning('NOTE: this check may have false positives or negatives depending on how parameters are referenced')
    all_obsolete = list()
    architecture_names = architectures.get_available_architectures()
    for idx_config, architecture_name in enumerate(architecture_names):
        config = configs.create_config_template(architecture_name)
        if idx_config == 0:
            all_obsolete.extend(_check_generic_config_parameters(config))
        all_obsolete.append(_check_architecture_config_parameters(config))
    _logger.info('Config parameter check complete')
    return all_obsolete


def _check_generic_config_parameters(config: configs.Config) -> List[str]:
    all_obsolete = list()
    for config_section in sections.get_config_sections():
        section_name = config_section.get_config_name_as_snake_case()
        _logger.debug('Check options for config section {}'.format(section_name))
        section = getattr(config, section_name)
        obsolete = [option_key for option_key in section.get_option_keys()
                    if not _is_option_key_in_package(option_key, section_name)]
        message = 'Config section {} has {} obsolete parameters'.format(section_name, len(obsolete))
        if obsolete:
            message += ':  {}'.format(', '.join(obsolete))
        _logger.info(message)
        all_obsolete.append(message)
    return all_obsolete


def _check_architecture_config_parameters(config: configs.Config) -> str:
    obsolete = [option_key for option_key in config.architecture.get_option_keys()
                if _is_option_key_in_architecture_module(option_key, config.model_training.architecture_name)]
    message = 'Architecture config section for architecture {} has {} obsolete parameters'.format(
        config.model_training.architecture_name, len(obsolete))
    if obsolete:
        message += ':  {}'.format(', '.join(obsolete))
    _logger.info(message)
    return message


def _is_option_key_in_package(option_key: str, config_section: str) -> bool:
    string = '{}.{}'.format(config_section, option_key)
    return _is_string_in_package(string, '*.py')


def _is_option_key_in_architecture_module(option_key: str, architecture_name: str) -> bool:
    return _is_string_in_package(option_key, 'architectures/{}.py'.format(architecture_name))


def _is_string_in_package(string: str, includes: str) -> bool:
    command = 'grep -rl --include={} "{}" {}'.format(includes, string, utils.DIR_PROJECT_ROOT)
    response = subprocess.run(shlex.split(command), stdout=subprocess.PIPE)
    filenames = [os.path.basename(filepath) for filepath in response.stdout.decode().split('\n') if filepath]
    if len(filenames) == 0:
        return False
    return True


if __name__ == '__main__':
    check_config_parameters_not_obsolete()