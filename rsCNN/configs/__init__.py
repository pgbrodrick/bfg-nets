import copy
import re
from typing import List
import yaml

from rsCNN.data_management import scalers


FILENAME_CONFIG = 'config.yaml'

# TODO:  improve names where necessary
# TODO:  informative comments for each option
# TODO:  check downstream that len raw filename lists match len scalers if len scalers > 1
# TODO:  add successful_data_save_file as constant in some script
# TODO:  data_save_name changed to dir_model_out


class BaseConfigSection(object):
    _field_defaults = NotImplemented

    def __init__(self) -> None:
        return

    def set_config_options(self, config_options: dict) -> None:
        for field_name, field_default in self._field_defaults:
            setattr(self, field_name, config_options.pop(field_name) or field_default)
        return

    def check_config_validity(self) -> List[str]:
        errors = list()
        for field_name, field_default, field_type in self._field_defaults:
            value_provided = getattr(self, field_name)
            if type(value_provided) is not field_type:
                errors.append('The value for {} must be an {} but a {} was provided:  {}'.format(
                    field_name, field_type, type(value_provided), value_provided))
        return errors


class RawFiles(BaseConfigSection):
    """
    Raw file configuration, information necessary to locate and parse the raw files.
    """
    raw_feature_file_list = None
    raw_response_file_list = None
    boundary_file_list = None
    feature_raw_band_type_input = None
    response_raw_band_type_input = None
    feature_nodata_value = None
    response_nodata_value = None
    boundary_bad_value = None
    ignore_projections = None
    # TODO:  expand on this
    # Data type from each feature.  R == Real, C == categorical
    # All Categorical bands will be one-hot-encoded...to keep them as a single band, simply name them as real datatypes
    _field_defaults = [
        ('raw_feature_file_list', None, list),  # File list for raw feature rasters, prior to being built
        ('raw_response_file_list', None, list),  # File list for raw response rasters, prior to being built
        ('boundary_file_list', None, list),  # Optional file list of boundaries, restricts footprint of built data
        ('feature_raw_band_type_input', None, str),  # See above note
        ('response_raw_band_type_input', None, str),  # See above note
        ('feature_nodata_value', -9999, float),  # Value that denotes missing data in feature files
        ('response_nodata_value', -9999, float),  # Value that denotes missing data in response files
        ('boundary_bad_value', None, float),  # Value that indicates pixels are out of bounds in a boundary raster file
        ('ignore_projections', False, bool),  # Ignore projection differences between feature/response, use with caution
    ]

    def check_config_validity(self) -> List[str]:
        errors = list()
        if type(self.raw_feature_file_list) is list and type(self.raw_response_file_list) is list:
            if len(self.raw_feature_file_list) == 0:
                errors.append('raw_feature_file_list must have more than one file')
            if len(self.raw_response_file_list) == 0:
                errors.append('raw_response_file_list must have more than one file')
            if len(self.raw_feature_file_list) != len(self.raw_response_file_list):
                errors.append('raw_feature_file_list and raw_response_file_list must have corresponding files and ' +
                              'be the same length')
        if self.raw_feature_file_list is None:
            errors.append('raw_feature_file_list must be provided')
        if self.raw_response_file_list is None:
            errors.append('raw_response_file_list must be provided')
        if type(self.boundary_file_list) is list:
            if self.boundary_bad_value is None:
                errors.append('boundary_bad_value must be provided if boundary_file_list is provided')
        return errors


class DataBuild(BaseConfigSection):
    """
    Data build configuration, information necessary to structure and format the built data files
    """
    dir_data_out = None
    response_data_format = None
    data_build_category = None
    data_save_name = None
    random_seed = None
    max_samples = None
    n_folds = None
    validation_fold = None
    test_fold = None
    window_radius = None
    internal_window_radius = None
    feature_nodata_maximum_fraction = None
    response_min_value = None
    response_max_value = None
    response_background_value = None
    field_defaults = [
        ('dir_data_out', None, str),  # Location to either create new built data files or load existing data files
        ('response_data_format', 'FCN', str),  # Either CNN or FCN right now
        ('data_build_category', 'or', str),  # TODO description
        ('data_save_name', None, str),  # Location to save the built data # TODO simplify
        ('random_seed', None, int),  # Seed to set for reproducable data generation
        ('max_samples', None, int),  # Max number of samples, sampling stops when data fully crawled or max is reached
        ('n_folds', 10, int),  # Number of training data folds
        ('validation_fold', None, int),  # Which training data fold to use for validation
        ('test_fold', None, int),  # Which training data fold to use for testing
        ('window_radius', None, int),  # Determines image size as 2 * window_radius
        ('internal_window_radius', None, int),  # Determines size of model loss window, must contain responses
        ('feature_nodata_maximum_fraction', 0.0, float),  # Only include samples where features have fewer nodata values
        ('response_min_value', None, float),  # Responses below this value are converted to the response_nodata_value
        ('response_max_value', None, float),  # Responses above this value are converted to the response_nodata_value
        ('response_background_value', None, float),  # Samples containing only this response will be discarded
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
    apply_random_transformations = None
    batch_size = None
    feature_scaler_names_list = None
    response_scaler_names_list = None
    # TODO:  Phil:  should mean_centering be a list so that we have one item per file?
    feature_mean_centering = None
    feature_training_nodata_value = None
    # Data sample configuration, information necessary to parse built data files and pass data to models during training
    field_defaults = [
        ('apply_random_transformations', False, bool),  # Whether to apply random rotations and flips to sample images
        ('batch_size', 100, int),  # The sample batch size for images passed to the model
        ('feature_scaler_names_list', None, list),  # Names of the scalers to use with each feature file
        ('response_scaler_names_list', None, list),  # Names of the scalers to use with each response file
        # TODO:  Phil:  should mean_centering be a list so that we have one item per file?
        ('feature_mean_centering', False, bool),  # Whether to mean center the features
        ('feature_training_nodata_value', -10.0, float),  # The missing data value for models, not compatible with nans
    ]

    def check_config_validity(self) -> List[str]:
        errors = super().check_config_validity()
        # TODO
        for scaler_name in self.feature_scaler_names_list:
            if not scalers.check_scaler_exists(scaler_name):
                errors.append('feature_scaler_names_list contains a scaler name that does not exist:  {}'.format(
                    scaler_name))
        for scaler_name in self.response_scaler_names_list:
            if not scalers.check_scaler_exists(scaler_name):
                errors.append('response_scaler_names_list contains a scaler name that does not exist:  {}'.format(
                    scaler_name))
        return errors


class ModelTraining(BaseConfigSection):
    dir_model_out = None
    verbosity = None
    assert_gpu = None
    architecture = None
    loss_metric = None
    max_epochs = None
    n_classes = None
    optimizer = None
    output_activation = None
    weighted = None
    # Model training configuration, information necessary to train models from start to finish
    field_defaults = [
        ('dir_model_out', None, str),  # Location to either create new model files or load existing model files
        ('verbosity', 1, int),  # Verbosity value for keras library, either 0 or 1
        ('assert_gpu', False, bool),  # Asserts GPU available before model training if True
        ('architecture', None, str),
        ('loss_metric', None, str),
        ('max_epochs', 100, int),
        ('weighted', False, bool),
    ]

    def check_config_validity(self) -> List[str]:
        errors = super().check_config_validity()
        # TODO
        return errors


class CallbackGeneral(BaseConfigSection):
    checkpoint_periods = None
    use_terminate_on_nan = None
    field_defaults = [
        ('checkpoint_periods', 5, int),
        ('use_terminate_on_nan', True, bool),
    ]


class CallbackTensorboard(BaseConfigSection):
    use_tensorboard = None
    dirname_prefix_tensorboard = None
    update_freq = None
    histogram_freq = None
    write_graph = None
    write_grads = None
    write_images = None
    _field_defaults = [
        ('use_tensorboard', True, bool),
        ('dirname_prefix_tensorboard', 'tensorboard', str),
        ('update_freq', 'epoch', str),
        ('histogram_freq', 0, int),
        ('write_graph', True, bool),
        ('write_grads', False, bool),
        ('write_images', True, bool),
    ]


class CallbackEarlyStopping(BaseConfigSection):
    use_early_stopping = None
    min_delta = None
    patience = None
    _field_defaults = [
        ('use_early_stopping', True),
        ('min_delta', 0.0001),
        ('patience', 50),
    ]


class CallbackReducedLearningRate(BaseConfigSection):
    use_reduced_learning_rate = None
    factor = None
    min_delta = None
    patience = None
    _field_defaults = [
        ('use_reduced_learning_rate', True, bool),
        ('factor', 0.5, float),
        ('min_delta', 0.0001, float),
        ('patience', 10, int),
    ]


def create_config_from_file(filepath: str) -> 'Config':
    with open(filepath) as file_:
        raw_config = yaml.safe_load(file_)
    config_factory = ConfigFactory()
    return config_factory.create_config(raw_config)


def create_config_template(filepath: str) -> None:
    config_factory = ConfigFactory()
    config = config_factory.create_config(dict())
    save_config_to_file(config, filepath)


def save_config_to_file(config: 'Config', filepath: str) -> None:
    config_out = dict()
    for section_name, config_section in config.__dict__.items():
        config_out[section_name] = {field: getattr(config_section, field) for field in config_section._fields}
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


class ConfigFactory(object):

    def __init__(self) -> None:
        return

    def create_config(self, config_options: dict) -> 'Config':
        config_sections = [
            RawFiles, DataBuild, DataSamples, ModelTraining, CallbackGeneral, CallbackTensorboard,
            CallbackEarlyStopping, CallbackReducedLearningRate
        ]
        populated_sections = dict()
        errors = list()
        config_copy = copy.deepcopy(config_options)
        for config_section in config_sections:
            section_name = self._convert_camelcase_to_snakecase(config_section.__name__)
            populated_section = self._create_config_section_from_options(config_copy, config_section)
            populated_sections[section_name] = populated_section
            errors.extend(populated_section.check_config_validity())
        if len(config_copy) > 0:
            errors.append('The configuration has unused options:  {}'.format(', '.join(list(config_copy.keys()))))
        assert len(errors) == 0, \
            '{} errors were found while building configuration:\n  {}'.format(len(errors), '\n'.join(errors))
        return Config(**populated_sections)

    def _create_config_section_from_options(self, config_options: dict, config_section: BaseConfigSection) \
            -> BaseConfigSection:
        values = dict()
        for field_name, _ in config_section._field_defaults:
            if field_name in config_options:
                values[field_name] = config_options.pop(field_name)
        config_section = config_section()
        config_section.set_config_options(config_options)
        return config_section

    def _convert_camelcase_to_snakecase(self, string: str) -> str:
        snake_case_converter = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return snake_case_converter.sub(r'_\1', string).lower()


class Config(object):
    raw_files = None
    data_build = None
    data_samples = None
    model_training = None
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
            callback_general: CallbackGeneral = None,
            callback_tensorboard: CallbackTensorboard = None,
            callback_early_stopping: CallbackEarlyStopping = None,
            callback_reduced_learning_rate: CallbackReducedLearningRate = None
    ) -> None:
        # Note:  it's undesireable to have so many parameters passed to the __init__ method, but I've chosen to write
        # it this way because we can use Python typing and modern IDEs to autocomplete all of the attributes and
        # subattributes in downstream scripts. For example, "config.a" will autocomplete to "config.architecture" and,
        # more importantly, "config.architecture.w" will autocomplete to "config.architecture.weighted". Without this
        # autocomplete feature, the programmer is required to know the names of individual options and due to the
        # nature of scientific computing and the number of parameters that can be configured, this becomes burdensome.
        self.raw_files = raw_files
        self.data_build = data_build
        self.data_sample = data_samples
        self.model_training = model_training
        self.callback_general = callback_general
        self.callback_tensorboard = callback_tensorboard
        self.callback_early_stopping = callback_early_stopping
        self.callback_reduced_learning_rate = callback_reduced_learning_rate
