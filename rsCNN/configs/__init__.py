from collections import OrderedDict
import copy
import os
from typing import List

import yaml

from rsCNN.configs.shared import BaseConfigSection, ConfigOption
from rsCNN.data_management import scalers
from rsCNN.networks import architectures
from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)

FILENAME_CONFIG = 'config.yaml'

# TODO:  add documentation for how to handle this file
# TODO:  check downstream that len raw filename lists match len scalers if len scalers > 1
# TODO:  add functions like get_available_architectures to get available options for all config options


class RawFiles(BaseConfigSection):
    """
    Raw file configuration, information necessary to locate and parse the raw files.
    """
    feature_files = None
    response_files = None
    boundary_files = None
    feature_data_type = None
    response_data_type = None
    feature_nodata_value = None
    response_nodata_value = None
    boundary_bad_value = None
    ignore_projections = None

    data_type_str = \
        'Data type from each input feature band.  R for Real, C for Categorical.  All C bands will be one-hot ' + \
        'encoded. Can be provided as a single string value(e.g. \'C\') or as a list of lists corresponding to each ' + \
        'band from each file in the raw input files list.'

    _config_options = [
        ConfigOption('feature_files', None, list, 'List of filepaths to raw feature rasters.'),
        ConfigOption('response_files', None, list, 'List of filepaths to raw response rasters.'),
        ConfigOption('boundary_files', None, list,
                     'Optional list of filepaths to boundaries. Data is built or sampled within the boundaries.'),
        ConfigOption('feature_data_type', None, str, data_type_str),  # See above note
        ConfigOption('response_data_type', None, str, data_type_str),  # See above note
        ConfigOption('feature_nodata_value', -9999, float, 'Value that denotes missing data in feature files.'),
        ConfigOption('response_nodata_value', -9999, float, 'Value that denotes missing data in response files.'),
        ConfigOption('boundary_bad_value', None, float, 'Value that denotes out-of-bounds areas in boundary files.'),
        ConfigOption('ignore_projections', False, bool,
                     'Should projection differences between feature and response files be ignored? This option ' +
                     'should only be true if the user is confident that projections are identical despite encodings.'),
    ]

    def check_config_validity(self) -> List[str]:
        errors = super().check_config_validity()
        if type(self.feature_files) is list and type(self.response_files) is list:
            if len(self.feature_files) == 0:
                errors.append('feature_files must have more than one file')
            if len(self.response_files) == 0:
                errors.append('response_files must have more than one file')
            if len(self.feature_files) != len(self.response_files):
                errors.append('feature_files and response_files must have corresponding files and ' +
                              'be the same length')
        if self.feature_files is None:
            errors.append('feature_files must be provided')
        if self.response_files is None:
            errors.append('response_files must be provided')
        if type(self.boundary_files) is list:
            if self.boundary_bad_value is None:
                errors.append('boundary_bad_value must be provided if boundary_files is provided')
        return errors


class DataBuild(BaseConfigSection):
    """
    Data build configuration, information necessary to structure and format the built data files
    """
    dir_out = None
    filename_prefix_out = None
    response_data_format = None
    random_seed = None
    max_samples = None
    number_folds = None
    validation_fold = None
    test_fold = None
    window_radius = None
    loss_window_radius = None
    # TODO:  Phil:  should mean_centering be a list so that we have one item per file?
    feature_mean_centering = None
    feature_nodata_maximum_fraction = None
    response_min_value = None
    response_max_value = None
    response_background_value = None
    _config_options = [
        ConfigOption('random_seed', 0, int, 'Random seed for reproducible data generation.'),
        ConfigOption('dir_out', None, str, 'Directory to which built data files are saved.'),
        ConfigOption('filename_prefix_out', None, str,
                     'Optional prefix for built data filenames, useful for organizing or tracking built data files ' +
                     'from different build strategies.'),
        # TODO:  rename the following?
        ConfigOption('response_data_format', 'FCN', str,
                     'Either CNN for convolutional neural network or FCN for fully convolutional network.'),
        ConfigOption('max_samples', None, int, 'Maximum number of built data samples to draw from the raw data ' +
                     'files. Sampling stops when the raw data files are fully crawled or the maximum samples are ' +
                     'reached.'),
        ConfigOption('window_radius', None, int,
                     'Window radius determines the full image size as 2x the window radius'),
        ConfigOption('loss_window_radius', None, int,
                     'Loss window radius determines the internal image window to use for loss calculations during ' +
                     'model training.'),
        ConfigOption('number_folds', 10, int, 'Number of training data folds.'),
        ConfigOption('validation_fold', 0, int, 'Index of fold to use for validation.'),
        ConfigOption('test_fold', 1, int, 'Index of fold to use for testing.'),
        # TODO:  Phil:  should mean_centering be a list so that we have one item per file?
        ConfigOption('feature_mean_centering', False, bool, 'Should features be mean centered?'),
        ConfigOption('feature_nodata_maximum_fraction', 0.0, float,
                     'Only include built data samples with a lower proportion of missing feature data values.'),

        # TODO: expand to multiple response values
        ConfigOption('response_min_value', None, float,
                     'Response values below this minimum are converted to missing data. Currently applied to all ' +
                     'response values uniformly.'),
        ConfigOption('response_max_value', None, float,
                     'Response values above this maximum are converted to missing data. Currently applied to all ' +
                     'response values uniformly.'),
        ConfigOption('response_background_value', None, float,
                     'Built data samples containing only this response are discarded and not included in the final ' +
                     'built data files.'),
    ]

    def check_config_validity(self) -> List[str]:
        errors = super().check_config_validity()
        # TODO
        response_data_format_options = ('FCN', 'CNN')
        if self.response_data_format not in response_data_format_options:
            errors.append('response_data_format is invalid option ({}), must be one of the following:  {}'.format(
                self.response_data_format, ','.join(response_data_format_options)
            ))
        return errors


class DataSamples(BaseConfigSection):
    """
    Data sample configuration, information necessary to parse built data files and pass data to models during training
    """
    apply_random_transformations = None
    batch_size = None
    feature_scaler_names = None
    response_scaler_names = None
    feature_nodata_encoding = None
    _config_options = [
        ConfigOption('apply_random_transformations', False, bool,
                     'Should random transformations, including rotations and flips, be applied to sample images.'),
        ConfigOption('batch_size', 100, int, 'The sample batch size for images passed to the model.'),
        ConfigOption('feature_scaler_names', None, list,
                     'Names of the scalers which are applied to each feature file.'),
        ConfigOption('response_scaler_names', None, list,
                     'Names of the scalers which are applied to each response file.'),
        ConfigOption('feature_nodata_encoding', -10.0, float,
                     'The encoding for missing data values passed to the model, given that neural networks are ' +
                     'sensitive to nans.'),
    ]

    def check_config_validity(self) -> List[str]:
        errors = super().check_config_validity()
        # TODO
        for scaler_name in self.feature_scaler_names:
            if not scalers.check_scaler_exists(scaler_name):
                errors.append('feature_scaler_names contains a scaler name that does not exist:  {}'.format(
                    scaler_name))
        for scaler_name in self.response_scaler_names:
            if not scalers.check_scaler_exists(scaler_name):
                errors.append('response_scaler_names contains a scaler name that does not exist:  {}'.format(
                    scaler_name))
        return errors


class ModelTraining(BaseConfigSection):
    """
    Model training configuration, information necessary to train models from start to finish
    """
    dir_out = None
    verbosity = None
    assert_gpu = None
    architecture_name = None
    loss_metric = None
    max_epochs = None
    optimizer = None
    weighted = None
    _config_options = [
        ConfigOption('dir_out', None, str,
                     'Directory to which new model files are saved and from which existing model files are loaded.'),
        ConfigOption('verbosity', 1, int, 'Verbosity value for keras library. Either 0 for silent or 1 for verbose.'),
        ConfigOption('assert_gpu', False, bool, 'Assert, i.e., fail if GPUs are required and not available.'),
        # TODO:  get imports working so you can autofill
        ConfigOption('architecture_name', None, str, 'Architecture name from existing options:  TODO.'),
        ConfigOption('loss_metric', None, str, 'Loss metric to use for model training.'),
        ConfigOption('max_epochs', 100, int, 'Maximum number of epochs to run model training.'),
        ConfigOption('optimizer', 'adam', str, 'Optimizer to use during model training.'),
        ConfigOption('weighted', False, bool, 'Should underrepresented classes be overweighted during model training'),
    ]

    def check_config_validity(self) -> List[str]:
        errors = super().check_config_validity()
        # TODO
        return errors


class CallbackGeneral(BaseConfigSection):
    checkpoint_periods = None
    use_terminate_on_nan = None
    _config_options = [
        ConfigOption('checkpoint_periods', 5, int,
                     'Number of periods of model training between model state and history saves.'),
        ConfigOption('use_terminate_on_nan', True, 'Terminate model training if nans are observed.'),
    ]


class CallbackTensorboard(BaseConfigSection):
    use_callback = None
    update_freq = None
    histogram_freq = None
    write_graph = None
    write_grads = None
    write_images = None
    _config_options = [
        ConfigOption('use_callback', True, bool,
                     'Use the Tensorboard callback to enable Tensorboard by saving necessary data.'),
        ConfigOption('update_freq', 'epoch', str, 'See keras documentation.'),
        ConfigOption('histogram_freq', 0, int, 'See keras documentation.'),
        ConfigOption('write_graph', True, bool, 'See keras documentation.'),
        ConfigOption('write_grads', False, bool, 'See keras documentation.'),
        ConfigOption('write_images', True, bool, 'See keras documentation.'),
    ]


class CallbackEarlyStopping(BaseConfigSection):
    use_callback = None
    min_delta = None
    patience = None
    _config_options = [
        ConfigOption('use_callback', True, bool, 'See keras documentation.'),
        ConfigOption('min_delta', 0.0001, float, 'See keras documentation.'),
        ConfigOption('patience', 50, int, 'See keras documentation.'),
    ]


class CallbackReducedLearningRate(BaseConfigSection):
    use_callback = None
    factor = None
    min_delta = None
    patience = None
    _config_options = [
        ConfigOption('use_callback', True, bool, 'See keras documentation.'),
        ConfigOption('factor', 0.5, float, 'See keras documentation.'),
        ConfigOption('min_delta', 0.0001, float, 'See keras documentation.'),
        ConfigOption('patience', 10, int, 'See keras documentation.'),
    ]


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


def compare_network_configs_get_differing_items(config_a, config_b):
    # TODO:  update for new classes
    differing_items = list()
    all_sections = set(list(config_a.keys()) + list(config_b.keys()))
    for section in all_sections:
        section_a = config_a.get(section, dict())
        section_b = config_b.get(section, dict())
        all_keys = set(list(section_a.keys()) + list(section_b.keys()))
        for key in all_keys:
            value_a = section_a.get(key, None)
            value_b = section_b.get(key, None)
            if value_a != value_b:
                differing_items.append((section, key, value_a, value_b))
    return differing_items


_CONFIG_SECTIONS = [
    RawFiles, DataBuild, DataSamples, ModelTraining, CallbackGeneral, CallbackTensorboard, CallbackEarlyStopping,
    CallbackReducedLearningRate
]


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
        for config_section in _CONFIG_SECTIONS:
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = config_section()
            populated_section.set_config_options(config_copy.get(section_name, dict()), is_template)
            populated_sections[section_name] = populated_section
        # Populate architecture options given architecture name
        architecture_name = populated_sections['model_training'].architecture_name
        architecture_options = architectures.get_architecture_options(architecture_name)
        architecture_options.set_config_options(config_copy.get('architecture_options', dict()), is_template)
        populated_sections['architecture_options'] = architecture_options
        return Config(**populated_sections)


class Config(object):
    raw_files = None
    data_build = None
    data_samples = None
    model_training = None
    architecture_options = None
    callback_general = None
    callback_tensorboard = None
    callback_early_stopping = None
    callback_reduced_learning_rate = None

    def __init__(
            self,
            raw_files: RawFiles = None,
            data_build: DataBuild = None,
            data_samples: DataSamples = None,
            model_training: ModelTraining = None,
            # TODO:  fix this reference
            architecture_options: 'BaseArchitectureOptions' = None,
            callback_general: CallbackGeneral = None,
            callback_tensorboard: CallbackTensorboard = None,
            callback_early_stopping: CallbackEarlyStopping = None,
            callback_reduced_learning_rate: CallbackReducedLearningRate = None
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
        self.architecture_options = architecture_options
        self.callback_general = callback_general
        self.callback_tensorboard = callback_tensorboard
        self.callback_early_stopping = callback_early_stopping
        self.callback_reduced_learning_rate = callback_reduced_learning_rate

    def get_config_as_dict(self) -> dict:
        config = OrderedDict()
        for config_section in _CONFIG_SECTIONS:
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            config[section_name] = populated_section.get_config_options_as_dict()
            if config_section is ModelTraining:
                # Given ordered output, architecture options make the most sense after model training options
                config['architecture_options'] = self.architecture_options.get_config_options_as_dict()
        return config

    def check_config_validity(self) -> bool:
        if self.get_config_errors():
            return False
        return True

    def get_config_errors(self) -> list:
        errors = list()
        for config_section in _CONFIG_SECTIONS:
            section_name = config_section.get_config_name_as_snake_case()
            populated_section = getattr(self, section_name)
            errors.extend(populated_section.check_config_validity())
            if config_section is ModelTraining:
                errors.extend(self.architecture_options.check_config_validity())
        return errors
