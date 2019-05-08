from collections import OrderedDict
import re
from typing import Dict, List

from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)


DEFAULT_HELP_TEXT = 'This config option needs help text for documentation purposes.'
DEFAULT_REQUIRED_VALUE = 'REQUIRED'


class ConfigOption(object):
    key = None
    value = None
    default = None
    type = None
    help_text = None

    def __init__(self, key: str, default: any, type: any, help_text: str = None) -> None:
        self.key = key
        self.default = default
        self.type = type
        self.help_text = help_text

    def get_help_text(self):
        if not self.help_text:
            return DEFAULT_HELP_TEXT
        return self.help_text


class BaseConfigSection(object):
    _config_options = NotImplemented

    def __init__(self) -> None:
        return

    @classmethod
    def get_config_name_as_snake_case(cls) -> str:
        snake_case_converter = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return snake_case_converter.sub(r'_\1', cls.__name__).lower()

    def get_config_options_as_dict(self) -> Dict[str, Dict[str, any]]:
        config_options = OrderedDict()
        for field in self.get_option_keys():
            value = getattr(self, field)
            if type(value) is tuple:
                value = list(value)  # Lists look nicer in config files and seem friendlier
            config_options[field] = value
        return config_options

    def get_option_keys(self) -> List[str]:
        return [option.key for option in self._config_options]

    def set_config_options(self, config_options: dict, highlight_required: bool) -> None:
        # TODO:  I expect an error here when reading in a config file and trying to parse that, we might need to
        #  automatically read the structure from different config sections and nestedness
        _logger.trace('Setting config options for section {} from {}'.format(self.__class__.__name__, config_options))
        for config_option in self._config_options:
            if config_option.key in config_options:
                option_value = config_options.pop(config_option.key)
            else:
                option_value = config_option.default
                if option_value is None and highlight_required:
                    option_value = DEFAULT_REQUIRED_VALUE
            setattr(self, config_option.key, option_value)
        return

    def check_config_validity(self) -> List[str]:
        errors = list()
        message_template = 'Invalid type for config option {} in config section {}. The provided value {} is a {}, ' + \
                           'but the required value should be a {}.'
        for config_option in self._config_options:
            value_provided = getattr(self, config_option.key)
            if type(value_provided) is not config_option.type:
                errors.append(message_template.format(
                    config_option.key, self.__class__.__name__, value_provided, type(value_provided), config_option.type
                ))
        return errors