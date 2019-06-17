from collections import OrderedDict
import logging
import re
from typing import Dict, List

from rsCNN.configuration import DEFAULT_OPTIONAL_VALUE, DEFAULT_REQUIRED_VALUE
from rsCNN.data_management import scalers


# TODO:  add documentation for how to handle this file
# TODO:  check downstream that len raw filename lists match len scalers if len scalers > 1

_logger = logging.getLogger(__name__)


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
        for key in self.get_option_keys():
            value = getattr(self, key)
            if type(value) is tuple:
                value = list(value)  # Lists look nicer in config files and seem friendlier
            config_options[key] = value
        return config_options

    def get_option_keys(self) -> List[str]:
        return [key for key in self.__class__.__dict__.keys()
                if not key.startswith('_') and not callable(getattr(self, key))]

    def set_config_options(self, config_options: dict, highlight_required: bool) -> None:
        _logger.debug('Setting config options for section {} from {}'.format(self.__class__.__name__, config_options))
        for key in self.get_option_keys():
            if key in config_options:
                value = config_options.pop(key)
                # YAML reads None in as a string
                if value in ('None', 'none'):
                    value = None
                _logger.debug('Setting option "{}" to provided value "{}"'.format(key, value))
            else:
                value = getattr(self, key)
                if not highlight_required and value in (DEFAULT_REQUIRED_VALUE, DEFAULT_OPTIONAL_VALUE):
                    value = None
                _logger.debug('Setting option "{}" to default value "{}"'.format(key, value))
            setattr(self, key, value)
        return

    def check_config_validity(self) -> List[str]:
        errors = list()
        message_template = 'Invalid type for config option {} in config section {}. The provided value {} is a {}, ' + \
                           'but the required value should be a {}.'
        for key in self.get_option_keys():
            value = getattr(self, key)
            type_expected = getattr(self, '_{}_type'.format(key))
            if type(value) is not type_expected:
                errors.append(message_template.format(key, self.__class__.__name__, value, type(value), type_expected))
        errors.extend(self._check_config_validity())
        return errors

    def _check_config_validity(self) -> List[str]:
        return list()


class RawFiles(BaseConfigSection):
    """
    Raw file configuration, information necessary to locate and parse the raw files.
    """
    _feature_files_type = list
    feature_files = DEFAULT_REQUIRED_VALUE
    """list: List of filepaths to raw feature rasters."""
    _response_files_type = list
    response_files = DEFAULT_REQUIRED_VALUE
    """list: List of filepaths to raw response rasters."""
    _boundary_files_type = list
    boundary_files = DEFAULT_OPTIONAL_VALUE
    """list: Optional list of filepaths to boundaries. Data is built or sampled within the boundaries."""
    _feature_data_type = str
    feature_data_type = DEFAULT_REQUIRED_VALUE
    """str: Data type from each input feature band.  R for Real, C for Categorical.  All C bands will be one-hot
    encoded. Can be provided as a single string value(e.g. \'C\') or as a list of lists corresponding to each
    band from each file in the raw input files list."""
    _response_data_type_type = str
    response_data_type = DEFAULT_REQUIRED_VALUE
    """str: Data type from each input feature band.  R for Real, C for Categorical.  All C bands will be one-hot
    encoded. Can be provided as a single string value(e.g. \'C\') or as a list of lists corresponding to each
    band from each file in the raw input files list."""
    _feature_nodata_value_type = float
    feature_nodata_value = -9999
    """float: Value that denotes missing data in feature files."""
    _response_nodata_value_type = float
    response_nodata_value = -9999
    """float: Value that denotes missing data in response files."""
    _boundary_bad_value_type = float
    boundary_bad_value = DEFAULT_OPTIONAL_VALUE
    """float: Value that denotes out-of-bounds areas in boundary files."""
    _ignore_projections_type = bool
    ignore_projections = False
    """bool: Should projection differences between feature and response files be ignored? This option ' +
    'should only be true if the user is confident that projections are identical despite encodings."""

    def _check_config_validity(self) -> List[str]:
        errors = list()
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
    _dir_out_type = str
    dir_out = '.'
    """str: Directory to which built data files are saved."""
    _filename_prefix_out_type = str
    filename_prefix_out = ''
    """str: Optional prefix for built data filenames, useful for organizing or tracking built data files
    from different build strategies."""
    # TODO:  rename the following?
    _response_data_format_type = str
    response_data_format = 'FCN'
    """str: Either CNN for convolutional neural network or FCN for fully convolutional network."""
    _random_seed_type = int
    random_seed = 1
    """int: Random seed for reproducible data generation."""
    _max_samples_type = int
    max_samples = DEFAULT_REQUIRED_VALUE
    """int: The maximum size of any given memmap array created in GB."""
    _max_memmap_size_gb = float
    max_memmap_size_gb = 10
    """int: Maximum number of built data samples to draw from the raw data files. Sampling stops when the raw data files 
    are fully crawled or the maximum samples are reached."""
    _number_folds_type = int
    number_folds = 10
    """int: Number of training data folds."""
    _validation_fold_type = int
    validation_fold = 0
    """int: Index of fold to use for validation."""
    _test_fold_type = int
    test_fold = DEFAULT_OPTIONAL_VALUE
    """int: Index of fold to use for testing."""
    _window_radius_type = int
    window_radius = DEFAULT_REQUIRED_VALUE
    """int: Window radius determines the full image size as 2x the window radius"""
    _loss_window_radius_type = int
    loss_window_radius = DEFAULT_REQUIRED_VALUE
    """int: Loss window radius determines the internal image window to use for loss calculations during model 
    training."""
    # TODO:  Phil:  should mean_centering be a list so that we have one item per file?
    _feature_mean_centering_type = bool
    feature_mean_centering = False
    """bool: Should features be mean centered?"""
    _feature_nodata_maximum_fraction_type = float
    feature_nodata_maximum_fraction = 0.0
    """float: Only include built data samples with a lower proportion of missing feature data values."""
    # TODO: expand to multiple response values
    _response_min_value_type = float
    response_min_value = DEFAULT_OPTIONAL_VALUE
    """float: Response values below this minimum are converted to missing data. Currently applied to all response values 
    uniformly."""
    # TODO: expand to multiple response values
    _response_max_value_type = float
    response_max_value = DEFAULT_OPTIONAL_VALUE
    """float: Response values above this maximum are converted to missing data. Currently applied to all response values 
    uniformly."""
    _response_background_value_type = int
    response_background_value = DEFAULT_OPTIONAL_VALUE
    """int: Built data samples containing only this response are discarded and not included in the final built data 
    files."""

    def _check_config_validity(self) -> List[str]:
        # TODO
        errors = list()
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
    _apply_random_transformations_type = bool
    apply_random_transformations = False
    """bool: Should random transformations, including rotations and flips, be applied to sample images."""
    _batch_size_type = int
    batch_size = 100
    """int: The sample batch size for images passed to the model."""
    _feature_scaler_names_type = list
    feature_scaler_names = DEFAULT_REQUIRED_VALUE
    """list: Names of the scalers which are applied to each feature file."""
    _response_scaler_names_type = list
    response_scaler_names = DEFAULT_REQUIRED_VALUE
    """list: Names of the scalers which are applied to each response file."""
    _feature_nodata_encoding_type = float
    feature_nodata_encoding = -10.0
    """float: The encoding for missing data values passed to the model, given that neural networks are sensitive to 
    nans."""

    def _check_config_validity(self) -> List[str]:
        # TODO
        errors = list()
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
    _dir_out_type = str
    dir_out = '.'
    """str: Directory to which new model files are saved and from which existing model files are loaded."""
    _verbosity_type = int
    verbosity = 1
    """int: Verbosity value for keras library. Either 0 for silent or 1 for verbose."""
    _assert_gpu_type = bool
    assert_gpu = False
    """bool: Assert, i.e., fail if GPUs are required and not available."""
    _architecture_name_type = str
    architecture_name = DEFAULT_REQUIRED_VALUE
    """str: Architecture name from existing options:  TODO."""
    _loss_metric_type = str
    loss_metric = DEFAULT_REQUIRED_VALUE
    """str: Loss metric to use for model training."""
    _max_epochs_type = int
    max_epochs = 100
    """int: Maximum number of epochs to run model training."""
    _optimizer_type = str
    optimizer = 'adam'
    """str: Optimizer to use during model training."""
    _weighted_type = bool
    weighted = False
    """bool: Should underrepresented classes be overweighted during model training"""

    def _check_config_validity(self) -> List[str]:
        # TODO
        errors = list()
        return errors


class CallbackGeneral(BaseConfigSection):
    _checkpoint_periods_type = int
    checkpoint_periods = 5
    """int: Number of periods of model training between model state and history saves."""
    _use_terminate_on_nan_type = bool
    use_terminate_on_nan = True
    """bool: Terminate model training if nans are observed."""


class CallbackTensorboard(BaseConfigSection):
    _use_callback_type = bool
    use_callback = True
    """bool: Use the Tensorboard callback to enable Tensorboard by saving necessary data."""
    _update_freq_type = str
    update_freq = "epoch"
    """str: See Keras documentation."""
    _histogram_freq_type = int
    histogram_freq = 0
    """int: See Keras documentation."""
    _write_graph_type = bool
    write_graph = True
    """bool: See Keras documentation."""
    _write_grads_type = bool
    write_grads = False
    """bool: See Keras documentation."""
    _write_images_type = bool
    write_images = True
    """bool: See Keras documentation."""


class CallbackEarlyStopping(BaseConfigSection):
    _use_callback_type = bool
    use_callback = True
    """bool: See Keras documentation."""
    _min_delta_type = float
    min_delta = 0.0001
    """float: See Keras documentation."""
    _patience_type = int
    patience = 50
    """int: See Keras documentation."""


class CallbackReducedLearningRate(BaseConfigSection):
    _use_callback_type = bool
    use_callback = True
    """bool: See Keras documentation."""
    _factor_type = float
    factor = 0.5
    """float: See Keras documentation."""
    _min_delta_type = float
    min_delta = 0.0001
    """float: See Keras documentation."""
    _patience_type = int
    patience = 10
    """int: See Keras documentation."""


def get_config_sections() -> List:
    return [
        RawFiles, DataBuild, DataSamples, ModelTraining, CallbackGeneral, CallbackTensorboard, CallbackEarlyStopping,
        CallbackReducedLearningRate
    ]
