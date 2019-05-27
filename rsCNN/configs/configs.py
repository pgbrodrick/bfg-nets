from collections import OrderedDict
import copy
import logging
import os

import yaml

import rsCNN.architectures.config_sections
import rsCNN.configs.sections


_logger = logging.getLogger(__name__)


FILENAME_CONFIG = 'config.yaml'


def create_config_from_file(dir_config: str, filename: str) -> 'Config':
    filepath = os.path.join(dir_config, filename or FILENAME_CONFIG)
    assert os.path.exists(filepath), 'No config file found at {}'.format(filepath)
    _logger.debug('Loading config file from {}'.format(filepath))
    with open(filepath) as file_:
        raw_config = yaml.safe_load(file_)
    config_factory = ConfigFactory()
    return config_factory.create_config(raw_config)


def create_config_template(architecture_name: str, dir_config: str, filename: str = None) -> None:
    _logger.debug('Creating config template for architecture {} at {}'.format(
        architecture_name, os.path.join(dir_config, filename)))
    config_factory = ConfigFactory()
    config = config_factory.create_config_template(architecture_name)
    save_config_to_file(config, dir_config, filename)


def save_config_to_file(config: 'Config', dir_config: str, filename: str = None, include_sections: list = None) -> None:
    def _represent_dictionary_order(self, dict_data):
        # via https://stackoverflow.com/questions/31605131/dumping-a-dictionary-to-a-yaml-file-while-preserving-order
        return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())

    def _represent_list_inline(self, list_data):
        return self.represent_sequence('tag:yaml.org,2002:seq', list_data, flow_style=True)

    yaml.add_representer(OrderedDict, _represent_dictionary_order)
    yaml.add_representer(list, _represent_list_inline)
    config_out = config.get_config_as_dict()
    filepath = os.path.join(dir_config, filename or FILENAME_CONFIG)
    _logger.debug('Saving config file to {}'.format(filepath))
    if include_sections:
        _logger.debug('Only saving config sections: {}'.format(', '.join(include_sections)))
        config_out = {section: config_out[section] for section in include_sections}
    with open(filepath, 'w') as file_:
        yaml.dump(config_out, file_, default_flow_style=False)


class ConfigFactory(object):

    def __init__(self) -> None:
        return

    def create_config(self, config_options: dict) -> 'Config':
        return self._create_config(config_options, is_template=False)

    def create_config_template(self, architecture_name: str) -> 'Config':
        config_options = {'model_training': {'architecture_name': architecture_name}}
        return self._create_config(config_options, is_template=True)

    def _create_config(self, config_options: dict, is_template: bool) -> 'Config':
        config_copy = copy.deepcopy(config_options)  # Use a copy because config options are popped from the dict
        # Population config sections with the provided configuration options, tracking errors
        populated_sections = dict()
        for config_section in rsCNN.configs.sections.get_config_sections():
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = config_section()
            populated_section.set_config_options(config_copy.get(section_name, dict()), is_template)
            populated_sections[section_name] = populated_section
        # Populate architecture options given architecture name
        architecture_name = populated_sections['model_training'].architecture_name
        architecture = rsCNN.architectures.config_sections.get_architecture_config_section(architecture_name)
        architecture.set_config_options(config_copy.get('architecture', dict()), is_template)
        populated_sections['architecture'] = architecture
        return Config(**populated_sections)


class Config(object):
    raw_files = None
    data_build = None
    data_samples = None
    model_training = None
    architecture = None
    callback_general = None
    callback_tensorboard = None
    callback_early_stopping = None
    callback_reduced_learning_rate = None

    def __init__(
            self,
            raw_files: rsCNN.configs.sections.RawFiles = None,
            data_build: rsCNN.configs.sections.DataBuild = None,
            data_samples: rsCNN.configs.sections.DataSamples = None,
            model_training: rsCNN.configs.sections.ModelTraining = None,
            architecture: rsCNN.architectures.config_sections.BaseArchitectureConfigSection = None,
            callback_general: rsCNN.configs.sections.CallbackGeneral = None,
            callback_tensorboard: rsCNN.configs.sections.CallbackTensorboard = None,
            callback_early_stopping: rsCNN.configs.sections.CallbackEarlyStopping = None,
            callback_reduced_learning_rate: rsCNN.configs.sections.CallbackReducedLearningRate = None
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
        self.callback_general = callback_general
        self.callback_tensorboard = callback_tensorboard
        self.callback_early_stopping = callback_early_stopping
        self.callback_reduced_learning_rate = callback_reduced_learning_rate

    def get_config_as_dict(self) -> dict:
        config = OrderedDict()
        for config_section in rsCNN.configs.sections.get_config_sections():
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            config[section_name] = populated_section.get_config_options_as_dict()
            if config_section is rsCNN.configs.sections.ModelTraining:
                # Given ordered output, architecture options make the most sense after model training options
                config['architecture'] = self.architecture.get_config_options_as_dict()
        return config

    def get_config_errors(self) -> list:
        errors = list()
        for config_section in rsCNN.configs.sections.get_config_sections():
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            errors.extend(populated_section.check_config_validity())
            if config_section is rsCNN.configs.sections.ModelTraining:
                errors.extend(self.architecture.check_config_validity())
        return errors
