from collections import OrderedDict
import copy
import logging
import os
from typing import Dict, List

import yaml

from rsCNN.architectures import config_sections
from rsCNN.configuration import sections


_logger = logging.getLogger(__name__)


DEFAULT_FILENAME_CONFIG = 'config.yaml'


class Config(object):
    """
    Handles the reading and formatting of raw data files, the building and training of models and architectures, and
    the reporting of training and validation results.
    """
    raw_files = None
    """sections.RawFiles: RawFiles config section."""
    data_build = None
    """sections.DataBuild: DataBuild config section."""
    data_samples = None
    """sections.DataSamples: DataSamples config section."""
    model_training = None
    """sections.ModelTraining: ModelTraining config section."""
    architecture = None
    """sections.Architecture: Architecture config section."""
    model_reporting = None
    """sections.ModelReporting: ModelReporting config section."""
    callback_general = None
    """sections.CallbacksGeneral: CallbacksGeneral config section."""
    callback_tensorboard = None
    """sections.Tensorboard: Tensorboard config section."""
    callback_early_stopping = None
    """sections.EarlyStopping: EarlyStopping config section."""
    callback_reduced_learning_rate = None
    """sections.CallBackReducedLearningRate: CallBackReducedLearningRate config section."""

    def __init__(
            self,
            raw_files: sections.RawFiles = None,
            data_build: sections.DataBuild = None,
            data_samples: sections.DataSamples = None,
            model_training: sections.ModelTraining = None,
            architecture: config_sections.BaseArchitectureConfigSection = None,
            model_reporting: sections.ModelReporting = None,
            callback_general: sections.CallbackGeneral = None,
            callback_tensorboard: sections.CallbackTensorboard = None,
            callback_early_stopping: sections.CallbackEarlyStopping = None,
            callback_reduced_learning_rate: sections.CallbackReducedLearningRate = None
    ) -> None:
        # Note:  it's undesireable to have so many parameters passed to the __init__ method and have so much boilerplate
        # code, but I've chosen to write it this way because we can use Python typing and modern IDEs to autocomplete
        # all of the attributes and subattributes in downstream scripts. For example, "config.a" will autocomplete to
        # "config.architecture" and, more importantly, "config.architecture.w" will autocomplete to
        # "config.architecture.weighted". Without this autocomplete feature, the programmer is required to know the
        # names of individual options and due to the nature of scientific computing and the number of parameters that
        # can be configured, this becomes burdensome.
        self.raw_files = raw_files
        self.data_build = data_build
        self.data_samples = data_samples
        self.model_training = model_training
        self.architecture = architecture
        self.model_reporting = model_reporting
        self.callback_general = callback_general
        self.callback_tensorboard = callback_tensorboard
        self.callback_early_stopping = callback_early_stopping
        self.callback_reduced_learning_rate = callback_reduced_learning_rate

    def get_config_as_dict(self) -> dict:
        """Get configuration options as a nested dictionary with delineated sections.

        Returns:
            Configuration options as a nested dictionary with delineated sections.
        """
        config = OrderedDict()
        for config_section in sections.get_config_sections():
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            config[section_name] = populated_section.get_config_options_as_dict()
            if config_section is sections.ModelTraining:
                # Given ordered output, architecture options make the most sense after model training options
                config['architecture'] = self.architecture.get_config_options_as_dict()
        return config

    def get_config_errors(self, include_sections: List[str] = None, exclude_sections: List[str] = None) -> list:
        """Get configuration option errors by checking the validity of each config section.

        Args:
            include_sections: Config sections that should be included. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.
            exclude_sections: Config sections that should be excluded. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.

        Returns:
            List of errors associated with the current configuration.
        """
        assert not (include_sections and exclude_sections), \
            'Both include_sections and exclude_sections cannot be specified.'
        _logger.debug('Checking config sections for errors')
        errors = list()
        config_sections = sections.get_config_sections()
        if include_sections:
            _logger.debug('Only checking config sections: {}'.format(', '.join(include_sections)))
            config_sections = [section for section in config_sections
                               if section.get_config_name_as_snake_case() in include_sections]
        if exclude_sections:
            _logger.debug('Not checking config sections: {}'.format(', '.join(exclude_sections)))
            config_sections = [section for section in config_sections
                               if section.get_config_name_as_snake_case() not in exclude_sections]
        for config_section in config_sections:
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            errors.extend(populated_section.check_config_validity())
            if config_section is sections.ModelTraining:
                errors.extend(self.architecture.check_config_validity())
        _logger.debug('{} errors found'.format(len(errors)))
        return errors

    def get_human_readable_config_errors(
            self,
            include_sections: List[str] = None,
            exclude_sections: List[str] = None
    ) -> str:
        """Generates a human-readable string of configuration option errors.

        Args:
            include_sections: Config sections that should be included. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.
            exclude_sections: Config sections that should be excluded. All config sections are included if None and
              exclude_sections is not specified. Cannot specify both include_sections and exclude_sections.

        Returns:
            Human-readable string of configuration option errors.
        """
        errors = self.get_config_errors(include_sections=include_sections, exclude_sections=exclude_sections)
        if not errors:
            return ''
        return 'List of configuration section and option errors is as follows:\n' + '\n'.join(error for error in errors)


def create_config_from_file(filepath: str) -> Config:
    """Creates a Config object from a YAML file.

    Args:
        filepath: Filepath to existing YAML file.

    Returns:
        Config object with parsed YAML file attributes.
    """
    assert os.path.exists(filepath), 'No config file found at {}'.format(filepath)
    _logger.debug('Loading config file from {}'.format(filepath))
    with open(filepath) as file_:
        raw_config = yaml.safe_load(file_)
    return _create_config(raw_config, is_template=False)


def create_config_template(architecture_name: str, filepath: str = None) -> Config:
    """Creates a template version of a Config for a given architecture, with required and optional parameters
    highlighted, and default values for other parameters. Config is returned but can optionally be written to YAML file.

    Args:
        architecture_name: Name of available architecture.
        filepath: Filepath to which template YAML file is saved, if desired.

    Returns:
        Template version of a Config.
    """
    _logger.debug('Creating config template for architecture {} at {}'.format(architecture_name, filepath))
    config_options = {'model_training': {'architecture_name': architecture_name}}
    config = _create_config(config_options, is_template=True)
    if filepath is not None:
        save_config_to_file(config, filepath)
    return config


def _create_config(config_options: dict, is_template: bool) -> Config:
    config_copy = copy.deepcopy(config_options)  # Use a copy because config options are popped from the dict
    # Populate config sections with the provided configuration options, tracking errors
    populated_sections = dict()
    for config_section in sections.get_config_sections():
        section_name = config_section.get_config_name_as_snake_case()
        populated_section = config_section()
        populated_section.set_config_options(config_copy.get(section_name, dict()), is_template)
        populated_sections[section_name] = populated_section
    # Populate architecture options given architecture name
    architecture_name = populated_sections['model_training'].architecture_name
    architecture = config_sections.get_architecture_config_section(architecture_name)
    architecture.set_config_options(config_copy.get('architecture', dict()), is_template)
    populated_sections['architecture'] = architecture
    return Config(**populated_sections)


def save_config_to_file(config: Config, filepath: str, include_sections: List[str] = None) -> None:
    """Saves/serializes a Config object to a YAML file.

    Args:
        config: Config object.
        filepath: Filepath to which YAML file is saved.
        include_sections: Config sections that should be included. All config sections are included if None.

    Returns:
        None
    """

    def _represent_dictionary_order(self, dict_data):
        # via https://stackoverflow.com/questions/31605131/dumping-a-dictionary-to-a-yaml-file-while-preserving-order
        return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())

    def _represent_list_inline(self, list_data):
        return self.represent_sequence('tag:yaml.org,2002:seq', list_data, flow_style=True)

    yaml.add_representer(OrderedDict, _represent_dictionary_order)
    yaml.add_representer(list, _represent_list_inline)
    config_out = config.get_config_as_dict()
    _logger.debug('Saving config file to {}'.format(filepath))
    if include_sections:
        _logger.debug('Only saving config sections: {}'.format(', '.join(include_sections)))
        config_out = {section: config_out[section] for section in include_sections}
    with open(filepath, 'w') as file_:
        yaml.dump(config_out, file_, default_flow_style=False)


def get_config_differences(config_a: Config, config_b: Config) -> Dict:
    differing_items = dict()
    dict_a = config_a.get_config_as_dict()
    dict_b = config_b.get_config_as_dict()
    all_sections = set(list(dict_a.keys()) + list(dict_b.keys()))
    for section in all_sections:
        section_a = dict_a.get(section, dict())
        section_b = dict_b.get(section, dict())
        all_options = set(list(section_a.keys()) + list(section_b.keys()))
        for option in all_options:
            if section == 'model_training' and option == 'dir_out':
                continue
            value_a = section_a.get(option, None)
            value_b = section_b.get(option, None)
            if value_a != value_b:
                _logger.debug('Configs have different values for option {} in section {}:  {} and {}'.format(
                    option, section, value_a, value_b))
                differing_items.setdefault(section, dict())[option] = (value_a, value_b)
    return differing_items
