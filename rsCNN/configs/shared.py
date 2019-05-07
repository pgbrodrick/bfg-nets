from collections import OrderedDict
import re
from typing import Dict, List

from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)


DEFAULT_REQUIRED_VALUE = 'REQUIRED'


class BaseConfigSection(object):
    _field_defaults = NotImplemented

    def __init__(self) -> None:
        return

    @classmethod
    def get_config_name_as_snake_case(cls) -> str:
        snake_case_converter = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return snake_case_converter.sub(r'_\1', cls.__name__).lower()

    def get_config_options_as_dict(self) -> Dict[str, Dict[str, any]]:
        config_options = OrderedDict()
        for field in self.get_fields():
            value = getattr(self, field)
            if type(value) is tuple:
                value = list(value)  # Lists look nicer in config files and seem friendlier
            config_options[field] = value
        return config_options

    def get_fields(self) -> List[str]:
        return [field for field, _, _ in self._field_defaults]

    def set_config_options(self, config_options: dict, highlight_required: bool) -> None:
        # TODO:  I expect an error here when reading in a config file and trying to parse that, we might need to
        #  automatically read the structure from different config sections and nestedness
        _logger.trace('Setting config options for {} from {}'.format(self.__class__.__name__, config_options))
        for field_name, field_default, _ in self._field_defaults:
            if field_name in config_options:
                field_value = config_options.pop(field_name)
            else:
                field_value = field_default
                if field_value is None and highlight_required:
                    field_value = DEFAULT_REQUIRED_VALUE
            setattr(self, field_name, field_value)
        return

    def check_config_validity(self) -> List[str]:
        errors = list()
        for field_name, field_default, field_type in self._field_defaults:
            value_provided = getattr(self, field_name)
            if type(value_provided) is not field_type:
                errors.append('The value for {} must be an {} but a {} was provided:  {}'.format(
                    field_name, field_type, type(value_provided), value_provided))
        return errors
