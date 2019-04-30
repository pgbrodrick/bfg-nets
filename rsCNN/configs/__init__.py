from collections import namedtuple
import copy
import re
import yaml


FILENAME_CONFIG = 'config.yaml'


field_defaults = [
    ('dir_out', None),
    ('verbosity', 1),
    ('assert_gpu', False),
]
Options = namedtuple('Options', [field for field, default in field_defaults])
Options.__new__.__defaults__ = tuple([default for field, default in field_defaults])

field_defaults = [
    ('architecture', None),
    ('inshape', None),
    ('internal_window_radius', None),
    ('loss_metric', None),
    ('n_classes', None),
    ('output_activation', None),
    ('weighted', False),
]
Architecture = namedtuple('Architecture', [field for field, default in field_defaults])
Architecture.__new__.__defaults__ = tuple([default for field, default in field_defaults])

field_defaults = [
    ('apply_random_transformations', False),
    ('max_epochs', 100),
    ('optimizer', 'adam'),
    ('batch_size', 100),
    ('modelfit_nan_value', -100),
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


def save_config_to_file(config: 'Config', filepath: str) -> None:
    config_out = dict()
    for section_name, config_section in config.__dict__.items():
        config_out[section_name] = {field: getattr(config_section, field) for field in config_section._fields}
    with open(filepath, 'w') as file_:
        yaml.dump(config_out, file_, default_flow_style=False)


def compare_network_configs_get_differing_items(config_a, config_b):
    # TODO:  update
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
    options = None
    architecture = None
    model_training = None
    callback_general = None
    callback_tensorboard = None
    callback_early_stopping = None
    callback_reduced_learning_rate = None

    def __init__(
            self,
            options: Options = None,
            architecture: Architecture = None,
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
        self.options = options
        self.architecture = architecture
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
            Options, Architecture, ModelTraining, CallbackGeneral, CallbackTensorboard, CallbackEarlyStopping,
            CallbackReducedLearningRate
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
