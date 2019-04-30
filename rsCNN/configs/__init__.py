from collections import namedtuple
import copy
import re
import yaml


FILENAME_CONFIG = 'config.yaml'

# TODO:  improve names where necessary
# TODO:  informative comments for each option

# TODO:  expand on this
# Data type from each feature.  R == Real, C == categorical
# All Categorical bands will be one-hot-encoded...to keep them as
# a single band, simply name them as real datatypes

# Raw file configuration, information necessary to locate and parse the raw files
field_defaults = [
    ('raw_feature_file_list', None),  # File list for raw feature rasters, prior to being built
    ('raw_response_file_list', None),  # File list for raw response rasters, prior to being built
    ('boundary_file_list', None),  # Optional list of boundary files, restricts built data to within these boundaries
    ('feature_raw_band_type_input', None),  # See above note
    ('response_raw_band_type_input', None),  # See above note
    ('feature_nodata_value', -9999),  # Value that denotes missing data in feature files
    ('response_nodata_value', -9999),  # Value that denotes missing data in response files
    ('boundary_bad_value', None),  # Value that indicates pixels are out of bounds in a boundary raster file
    ('ignore_projections', False),  # To ignore projection differences between feature/response sets, use with caution!
]
RawFiles = namedtuple('RawFiles', [field for field, default in field_defaults])
RawFiles.__new__.__defaults__ = tuple([default for field, default in field_defaults])

# Data build configuration, information necessary to structure and format the built data files
field_defaults = [
    ('dir_model_out', None),  # Location to either create new model files or load existing model files
    ('response_data_format', 'FCN'),  # Either CNN or FCN right now
    ('data_build_category', 'or'),  # TODO description
    ('data_save_name', None),  # Location to save the built data, including potential prefix for names # TODO simplify
    ('random_seed', None),  # Seed to set for reproducable data generation
    ('max_samples', None),  # Maximum number of samples, sampling stops when data is fully crawled or maximum is reached
    ('n_folds', 10),  # Number of training data folds
    ('validation_fold', None),  # Which training data fold to use for validation
    ('test_fold', None),  # Which training data fold to use for testing
    ('window_radius', None),  # Determines image size as 2 * window_radius
    ('internal_window_radius', None),  # Determines image subset included in model loss window, must contain responses
    ('feature_nodata_maximum_fraction', 0),  # Only include samples where features have a lower nodata fraction
    ('response_min_value', None),  # Responses below this value will be converted to the response_nodata_value
    ('response_max_value', None),  # Responses above this value will be converted to the response_nodata_value
    ('response_background_value', None),  # Samples containing only this response will be discarded
]
DataBuild = namedtuple('DataBuild', [field for field, default in field_defaults])
DataBuild.__new__.__defaults__ = tuple([default for field, default in field_defaults])

# Data sample configuration, information necessary to parse built data files and pass data to models during training
field_defaults = [
    ('apply_random_transformations', False),  # Whether to apply random rotations and flips to sample images
    ('batch_size', 100),  # The sample batch size for images passed to the model
    ('feature_scaler_names', None),  # Names of the scalers to use with each feature file
    ('response_scaler_names', None),  # Names of the scalers to use with each response file
    # TODO:  Phil:  should mean_centering be a list so that we have one item per file?
    ('feature_mean_centering', False),  # Whether to mean center the features
    ('feature_training_nodata_value', -10),  # The value for missing data, as models are not compatible with nans
]
DataSamples = namedtuple('DataSamples', [field for field, default in field_defaults])
DataSamples.__new__.__defaults__ = tuple([default for field, default in field_defaults])

field_defaults = [
    ('dir_data_out', None),  # Location to either create new built data files or load existing data files
    ('verbosity', 1),  # Verbosity value for keras library, either 0 or 1
    ('assert_gpu', False),  # Asserts that model will run on GPU if True, exits with error if GPU is not available
    ('architecture', None),
    ('loss_metric', None),
    ('max_epochs', 100),
    ('n_classes', None),  # TODO:  get this from the data directly
    ('optimizer', 'adam'),
    ('output_activation', None),
    ('weighted', False),
]
ModelTraining = namedtuple('ModelTraining', [field for field, default in field_defaults])
ModelTraining.__new__.__defaults__ = tuple([default for field, default in field_defaults])

field_defaults = [
    ('checkpoint_periods', 5),
    ('use_terminate_on_nan', True),
]
CallbackGeneral = namedtuple('CallbackGeneral', [field for field, default in field_defaults])
CallbackGeneral.__new__.__defaults__ = tuple([default for field, default in field_defaults])

field_defaults = [
    ('use_tensorboard', True),
    ('dirname_prefix_tensorboard', 'tensorboard'),
    ('update_freq', 'epoch'),
    ('histogram_freq', 0),
    ('write_graph', True),
    ('write_grads', False),
    ('write_images', True),
]
CallbackTensorboard = namedtuple('CallbackTensorboard', [field for field, default in field_defaults])
CallbackTensorboard.__new__.__defaults__ = tuple([default for field, default in field_defaults])

field_defaults = [
    ('use_early_stopping', True),
    ('min_delta', 0.0001),
    ('patience', 50),
]
CallbackEarlyStopping = namedtuple('CallbackEarlyStopping', [field for field, default in field_defaults])
CallbackEarlyStopping.__new__.__defaults__ = tuple([default for field, default in field_defaults])

field_defaults = [
    ('use_reduced_learning_rate', True),
    ('factor', 0.5),
    ('min_delta', 0.0001),
    ('patience', 10),
]
CallbackReducedLearningRate = namedtuple('CallbackReducedLearningRate', [field for field, default in field_defaults])
CallbackReducedLearningRate.__new__.__defaults__ = tuple([default for field, default in field_defaults])


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


class ConfigFactory(object):

    def __init__(self) -> None:
        return

    def create_config(self, config_options: dict) -> Config:
        config_copy = copy.deepcopy(config_options)
        config_sections = [
            RawFiles, DataBuild, DataSamples, ModelTraining, CallbackGeneral, CallbackTensorboard,
            CallbackEarlyStopping, CallbackReducedLearningRate
        ]
        populated_sections = dict()
        for config_section in config_sections:
            section_name = self._convert_camelcase_to_snakecase(config_section.__name__)
            populated_sections[section_name] = self._create_namedtuple_from_options(config_copy, config_section)
        assert not config_copy, 'The configuration has unused options:  {}'.format(', '.join(list(config_copy.keys())))
        return Config(**populated_sections)

    def _create_namedtuple_from_options(self, config_options: dict, config_section: namedtuple) -> namedtuple:
        values = dict()
        for field in config_section._fields:
            if field in config_options:
                values[field] = config_options.pop(field)
        return config_section(**values)

    def _convert_camelcase_to_snakecase(self, string: str) -> str:
        snake_case_converter = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return snake_case_converter.sub(r'_\1', string).lower()
