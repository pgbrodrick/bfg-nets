from collections import OrderedDict
import gdal
import logging
import numpy as np
import os
import re
from typing import Dict, List, Type

from rsCNN import architectures
from rsCNN.configuration import DEFAULT_OPTIONAL_VALUE, DEFAULT_REQUIRED_VALUE
from rsCNN.data_management import scalers
from rsCNN.experiments import losses


"""
How to use sections:  Configs and ConfigSections are the tools we (currently) use to handle the numerous parameters
associated with neural network experiments. In general, we want to validate that ConfigSections have options with
the expected values and relationships, and we use two important methods to do this without duplicating too much code.

Important note:  please feel free to suggest better ways of handling these things.

To validate that attributes have the correct type, ensure that the attributes are on the config section and have an
associated hidden attribute with a particular name pattern. Specifically, given an attribute named 'attribute', the
hidden attribute should be named '_attribute_type' and its value should be the type expected for that attribute.
Methods on the BaseConfigSection will ensure that this attribute type is checked and errors will be raised to the user
if it's not appropriate. Example:

```
class GenericConfigSection(BaseConfigSection):
    _attribute_type = list                  <-- Used to validate attributes have correct types
    attribute = DEFAULT_REQUIRED_VALUE
```

To validate that attributes have appropriate relationships or characteristics, use the hidden _check_config_validity
method to add more detailed validation checks. Simply return a list of string descriptions of errors from the method as
demonstrated:

```
def _check_config_validity(self) -> List[str]:
    errors = list()
    if self.attribute_min >= self.attribute_max:
        errors.append('attribute_min must be less than attribute_max.')
    return errors
```
"""


_logger = logging.getLogger(__name__)

VECTORIZED_FILENAMES = ('kml', 'shp')


class BaseConfigSection(object):
    """
    Base Configuration Section from which all Configuration Sections inherit. Handles shared functionality like getting,
    setting, and cleaning configuration options.
    """
    _config_options = NotImplemented
    _options_required = None
    _options_optional = None

    def __init__(self) -> None:
        self._options_required = [key for key in self.get_option_keys() if getattr(self, key) == DEFAULT_REQUIRED_VALUE]
        self._options_optional = [key for key in self.get_option_keys() if getattr(self, key) == DEFAULT_OPTIONAL_VALUE]

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
                _logger.debug('Setting option "{}" to provided value "{}"'.format(key, value))
            else:
                value = getattr(self, key)
                # We leave the labels for required and optional values as-is for templates, i.e., when highlights are
                # required, otherwise we convert those labels to None
                if not highlight_required and value in (DEFAULT_REQUIRED_VALUE, DEFAULT_OPTIONAL_VALUE):
                    value = None
                _logger.debug('Setting option "{}" to default value "{}"'.format(key, value))
            setattr(self, key, self._clean_config_option_value(key, value))
        return

    def _clean_config_option_value(self, option_key: str, value: any) -> any:
        # PyYAML reads None as a string so we need to convert to the None type
        if value in ('None', 'none'):
            value = None
        # Some parameters are treated as floats, but ints are acceptable input formats
        # Treating ints as floats is more flexible and requires fewer assumptions / accomodations in the code
        # Example:  users are likely to provide -9999 instead of -9999.0
        type_expected = self._get_expected_type_for_option_key(option_key)
        if type(value) is int and type_expected is float:
            value = float(value)
        # Some parameters are treated as tuples, but lists are acceptable input formats
        # Treating lists as tuples is more desirable for objects that should not mutable
        # Example:  YAML supports lists naturally, but requires more complex syntax for tuples, so users are not
        # likely to provide tuples
        if type(value) is list and type_expected is tuple:
            value = tuple(value)
        return value

    def check_config_validity(self) -> List[str]:
        errors = list()
        message_required = '{} in config section {} is required, but either 1) None was provided or 2) no value ' + \
                           'was provided. The expected type is {}.'
        message_type = 'Invalid type for config option {} in config section {}. The provided value {} is a {}, ' + \
                       'but the required value should be a {}.'
        for key in self.get_option_keys():
            value = getattr(self, key)
            # No further checking is necessary if the provided type matches to expected type
            type_expected = self._get_expected_type_for_option_key(key)
            if type(value) is type_expected:
                continue
            # Optional values which were not provided (i.e., are None), are acceptable
            if value is None and key in self._options_optional:
                continue
            # At this point, we know there's an issue with this key
            # Either we have a required option that can never be None, or we have a type mismatch
            if value is None and key in self._options_required:
                errors.append(message_required.format(key, self.__class__.__name__, type_expected))
            else:
                errors.append(message_type.format(key, self.__class__.__name__, value, type(value), type_expected))
        errors.extend(self._check_config_validity())
        return errors

    def _check_config_validity(self) -> List[str]:
        return list()

    def _get_expected_type_for_option_key(self, option_key: str) -> type:
        return getattr(self, '_{}_type'.format(option_key))


class RawFiles(BaseConfigSection):
    """
    Raw file configuration, information necessary to locate and parse the raw files.
    """
    _feature_files_type = list
    feature_files = DEFAULT_REQUIRED_VALUE
    """list: List of filepaths to raw feature rasters.  Format is a list of lists, where the outer
    grouping is sites, and the inner grouping is different data files for that particular site.
    E.G.: [[site_1_file_1, site_1_file_2],[site_2_file_1, site_2_file_2]].  File order and type
    between sites is expected to match.  Files must all be the same projection and resolution, but
    need not be aligned to the same extent - if they are not, the common inner-area will be
    utilized."""
    _response_files_type = list
    response_files = DEFAULT_REQUIRED_VALUE
    """list: List of filepaths to raw response rasters or vectors (shp or kml supported currently).  Format is a list of
    lists, where the outer grouping is sites, and the inner grouping is different data files for that particular site.
    E.G.: [[site_1_file_1, site_1_file_2],[site_2_file_1, site_2_file_2]].  File order and type between sites is 
    expected to match.  Files must all be the same projection and resolution, but need not be aligned to the same extent
    - if they are not, the common inner-area will be utilized."""
    _boundary_files_type = list
    boundary_files = DEFAULT_OPTIONAL_VALUE
    """list: Optional list of filepaths to boundaries. Data is built or sampled within the boundaries."""
    _feature_data_type_type = list
    feature_data_type = DEFAULT_REQUIRED_VALUE
    """list: Data type from each input feature band.  R for Real, C for Categorical.  All C bands will be one-hot
    encoded. Provided as a list of lists corresponding to each band from each file in the raw input files list."""
    _response_data_type_type = list
    response_data_type = DEFAULT_REQUIRED_VALUE
    """list: Data type from each input feature band.  R for Real, C for Categorical.  All C bands will be one-hot
    encoded. Provided as a list of lists corresponding to each band from each file in the raw input files list."""
    _feature_nodata_value_type = float
    feature_nodata_value = -9999
    """float: Value that denotes missing data in feature files."""
    _response_nodata_value_type = float
    response_nodata_value = -9999
    """float: Value that denotes missing data in response files."""
    _boundary_bad_value_type = float
    boundary_bad_value = DEFAULT_OPTIONAL_VALUE
    """float: Value that denotes out-of-bounds areas in boundary files. For example, raster files may have non-negative
    values to represent in-bounds areas and -9999 to represent out-of-bounds areas. This value must be 0 when working
    with shapefiles where vectors/polygons denote in-bounds areas."""
    _ignore_projections_type = bool
    ignore_projections = False
    """bool: Should projection differences between feature and response files be ignored? This option
    should only be true if the user is confident that projections are identical despite encodings."""

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
        if type(self.boundary_files) is list:
            if self.boundary_bad_value is None:
                errors.append('boundary_bad_value must be provided if boundary_files is provided')

        errors.extend(_check_input_file_formats(self.feature_files, self.response_files, self.boundary_files))

        return errors


class DataBuild(BaseConfigSection):
    """
    Data build configuration, information necessary to structure and format the built data files.
    """
    _dir_out_type = str
    dir_out = '.'
    """str: Directory to which built data files are saved."""
    _log_level_type = str
    log_level = 'INFO'
    """str: Experiment log level. One of ERROR, WARNING, INFO, or DEBUG."""
    _filename_prefix_out_type = str
    filename_prefix_out = ''
    """str: Optional prefix for built data filenames, useful for organizing or tracking built data files
    from different build strategies."""
    _network_category_type = str
    network_category = 'FCN'
    """str: Either CNN for convolutional neural network or FCN for fully convolutional network."""
    _random_seed_type = int
    random_seed = 1
    """int: Random seed for reproducible data generation."""
    _max_samples_type = int
    max_samples = DEFAULT_REQUIRED_VALUE
    """int: Maximum number of built data samples to draw from the raw data files. Sampling stops when the raw data files 
    are fully crawled or the maximum samples are reached."""
    _max_built_data_gb_type = float
    max_built_data_gb = 10.0
    """float: The maximum size of any given memmap array created in GB."""
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
    # TODO:  convert to a list, one instance per file
    _feature_mean_centering_type = bool
    feature_mean_centering = False
    """bool: Should features be mean centered?"""
    _feature_nodata_maximum_fraction_type = float
    feature_nodata_maximum_fraction = 0.0
    """float: Only include built data samples with a lower proportion of missing feature data values."""
    # TODO: Phil:  expand to multiple response values per file?
    _response_min_value_type = float
    response_min_value = DEFAULT_OPTIONAL_VALUE
    """float: Response values below this minimum are converted to missing data. Currently applied to all response values 
    uniformly."""
    # TODO: Phil:  expand to multiple response values per file?
    _response_max_value_type = float
    response_max_value = DEFAULT_OPTIONAL_VALUE
    """float: Response values above this maximum are converted to missing data. Currently applied to all response values 
    uniformly."""
    _response_background_value_type = int
    response_background_value = DEFAULT_OPTIONAL_VALUE
    """int: Built data samples containing only this response are discarded and not included in the final built data 
    files."""

    def _check_config_validity(self) -> List[str]:
        errors = list()
        response_data_format_options = ('FCN', 'CNN')
        if self.network_category not in response_data_format_options:
            errors.append('response_data_format is invalid option ({}), must be one of the following:  {}'.format(
                self.network_category, ','.join(response_data_format_options)
            ))
        return errors


class DataSamples(BaseConfigSection):
    """
    Data sample configuration, information necessary to parse built data files and pass data to models during training.
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
        errors = list()
        available_scalers = scalers.get_available_scalers()
        if (self.feature_scaler_names is list):
            for scaler_name in self.feature_scaler_names:
                if scaler_name not in available_scalers:
                    errors.append('feature_scaler_names contains a scaler name that does not exist:  {}'.format(
                        scaler_name))
        if (self.response_scaler_names is list):
            for scaler_name in self.response_scaler_names:
                if scaler_name not in available_scalers:
                    errors.append('response_scaler_names contains a scaler name that does not exist:  {}'.format(
                        scaler_name))
        return errors


class ModelTraining(BaseConfigSection):
    """
    Model training configuration, information necessary to train models from start to finish.
    """
    _dir_out_type = str
    dir_out = '.'
    """str: Directory to which new model files are saved and from which existing model files are loaded."""
    _log_level_type = str
    log_level = 'INFO'
    """str: Experiment log level. One of ERROR, WARNING, INFO, or DEBUG."""
    _verbosity_type = int
    verbosity = 1
    """int: Verbosity value for keras library. Either 0 for silent or 1 for verbose."""
    _assert_gpu_type = bool
    assert_gpu = False
    """bool: Assert, i.e., fail if GPUs are required and not available."""
    _architecture_name_type = str
    architecture_name = DEFAULT_REQUIRED_VALUE
    """str: Architecture name from existing options."""
    _loss_metric_type = str
    loss_metric = DEFAULT_REQUIRED_VALUE
    """str: Loss metric to use for model training."""
    _max_epochs_type = int
    max_epochs = 100
    """int: Maximum number of epochs to run model training."""
    _optimizer_type = str
    optimizer = 'adam'
    """str: Optimizer to use during model training. See Keras documentation for more information."""
    _weighted_type = bool
    weighted = False
    """bool: Should underrepresented classes be overweighted during model training"""

    def _check_config_validity(self) -> List[str]:
        errors = list()
        available_architectures = architectures.get_available_architectures()
        if self.architecture_name not in available_architectures:
            errors.append('architecture_name {} not in available architectures: {}'.format(
                self.architecture_name, available_architectures))
        available_losses = losses.get_available_loss_methods()
        if self.loss_metric not in available_losses:
            errors.append('loss_metric {} not in available loss metrics: {}'.format(self.loss_metric, available_losses))
        return errors


class ModelReporting(BaseConfigSection):
    """
    Model reporting configuration, information necessary to generate reports for evaluating model performance.
    """
    _max_pages_per_figure_type = int
    max_pages_per_figure = 1
    """int: The max number of pages per figure in the model report."""
    _max_samples_per_page_type = int
    max_samples_per_page = 20
    """int: The max number of samples per page in supported figures in the model report"""
    _max_features_per_page_type = int
    max_features_per_page = 10
    """int: The max number of features per page in supported figures in the model report"""
    _max_responses_per_page_type = int
    max_responses_per_page = 10
    """int: The max number of responses per page in supported figures in the model report"""
    _network_progression_max_pages_type = int
    network_progression_max_pages = 1
    """int: The max number of pages for the network progression figure in the model report. Note that the network
    progression figure is particularly expensive, both for computation and memory."""
    _network_progression_max_filters_type = int
    network_progression_max_filters = 10
    """int: The max number of filters for the network progression figure in the model report. Note that the network
    progression figure is particularly expensive, both for computation and memory."""
    _network_progression_show_full_type = bool
    network_progression_show_full = True
    """bool: Whether to plot the full network progression plot, with fully-visible filters."""
    _network_progression_show_compact_type = bool
    network_progression_show_compact = True
    """bool: Whether to plot the compact network progression plot, with partially-visible filters."""


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
    _loss_metric_type = str
    loss_metric = 'val_loss'
    """str: Loss metric to monitor for early stopping. See Keras documentation."""
    _min_delta_type = float
    min_delta = 0.0001
    """float: See Keras documentation."""
    _patience_type = int
    patience = 10
    """int: See Keras documentation."""


class CallbackReducedLearningRate(BaseConfigSection):
    _use_callback_type = bool
    use_callback = True
    """bool: See Keras documentation."""
    _factor_type = float
    factor = 0.5
    """float: See Keras documentation."""
    _loss_metric_type = str
    loss_metric = 'val_loss'
    """str: Loss metric to monitor for reducing learning rate. See Keras documentation."""
    _min_delta_type = float
    min_delta = 0.0001
    """float: See Keras documentation."""
    _patience_type = int
    patience = 5
    """int: See Keras documentation."""


def get_config_sections() -> List[Type[BaseConfigSection]]:
    return [
        RawFiles, DataBuild, DataSamples, ModelTraining, ModelReporting, CallbackGeneral, CallbackTensorboard,
        CallbackEarlyStopping, CallbackReducedLearningRate
    ]


def check_input_file_validity(f_file_list, r_file_list, b_file_list) -> List[str]:
    errors = _check_input_file_formats(f_file_list, r_file_list, b_file_list)
    # Checks that all files can be opened by gdal
    for _site in range(len(f_file_list)):
        for _band in range(len(f_file_list[_site])):
            if (gdal.Open(f_file_list[_site][_band], gdal.GA_ReadOnly) is None):
                errors.append('Could not open feature site {}, file {}'.format(_site, _band))

    for _site in range(len(r_file_list)):
        for _band in range(len(r_file_list[_site])):
            if(_noerror_open(r_file_list[_site][_band]) is None and
                    r_file_list[_site][_band].split('.')[-1] not in VECTORIZED_FILENAMES):
                errors.append('Could not open response site {}, file {}'.format(_site, _band))

    # Checks on the number of bands per file
    num_f_bands_per_file = [gdal.Open(x, gdal.GA_ReadOnly).RasterCount for x in f_file_list[0]]
    for _site in range(len(f_file_list)):
        for _file in range(len(f_file_list[_site])):
            if(gdal.Open(f_file_list[_site][_file], gdal.GA_ReadOnly).RasterCount != num_f_bands_per_file[_file]):
                errors.append('Inconsistent number of feature bands in site {}, file {}'.format(_site, _file))

    file_type = []
    for _site in range(len(r_file_list)):
        if (r_file_list[_site][0].split('.')[-1] in VECTORIZED_FILENAMES):
            file_type.append('V')
        else:
            file_type.append('R')

        for _file in range(1, len(r_file_list[0])):
            if (r_file_list[_site][_file].split('.')[-1] in VECTORIZED_FILENAMES):
                if (file_type[-1] == 'R'):
                    file_type[-1] = 'M'
                    break
            else:
                if (file_type[-1] == 'V'):
                    file_type[-1] = 'M'
                    break

    un_file_types = np.unique(file_type)
    if (len(un_file_types) > 1):
        errors.append('Response file types mixed, found per-site order:\n{}\nR=Raster\nV=Vector\nM=Mixed')

    if (np.all(un_file_types == 'R')):

        is_vector = any([x.split('.')[-1] in VECTORIZED_FILENAMES for x in r_file_list[0]])
        if not is_vector:
            num_r_bands_per_file = [gdal.Open(x, gdal.GA_ReadOnly).RasterCount for x in r_file_list[0]]
            for _site in range(len(r_file_list)):
                for _file in range(len(r_file_list[_site])):
                    if(gdal.Open(r_file_list[_site][_file], gdal.GA_ReadOnly).RasterCount != num_r_bands_per_file[_file]):
                        errors.append('Inconsistent number of response bands in site {}, file {}'.format(_site, _file))

    return errors


def _check_input_file_formats(f_file_list, r_file_list, b_file_list) -> List[str]:
    errors = []
    # f = feature, r = response, b = boundary

    # file lists r and f are expected a list of lists.  The outer list is a series of sites (location a, b, etc.).
    # The inner list is a series of files associated with that site (band x, y, z).  Each site must have the
    # same number of files, and each file from each site must have the same number of bands, in the same order.
    # file list b is a list for each site, with one boundary file expected to be the interior boundary for all
    # bands.

    # Check that feature and response files are lists
    if (type(f_file_list) is not list):
        errors.append('Feature files must be a list of lists')
    if (type(r_file_list) is not list):
        errors.append('Response files must be a list of lists')

    if (len(errors) > 0):
        errors.append('Feature or response files not in correct format...all checks cannot be completed')
        return errors

    # Checks on the matching numbers of sites
    if (len(f_file_list) != len(r_file_list)):
        errors.append('Feature and response site lists must be the same length')
    if (len(f_file_list) <= 0):
        errors.append('At least one feature and response site is required')
    if b_file_list is not None:
        if (len(b_file_list) != len(f_file_list)):
            errors.append(
                'Boundary and feature file lists must be the same length. Boundary list: {}. Feature list: {}.'.format(
                    b_file_list, f_file_list))

    # Checks that we have lists of lists for f and r
    for _site in range(len(f_file_list)):
        if(type(f_file_list[_site]) is not list):
            errors.append('Features at site {} are not as a list'.format(_site))

    for _site in range(len(r_file_list)):
        if(type(r_file_list[_site]) is not list):
            errors.append('Responses at site {} are not as a list'.format(_site))

    # Checks on the number of files per site
    num_f_files_per_site = len(f_file_list[0])
    num_r_files_per_site = len(r_file_list[0])
    for _site in range(len(f_file_list)):
        if(len(f_file_list[_site]) != num_f_files_per_site):
            errors.append('Inconsistent number of feature files at site {}'.format(_site))

    for _site in range(len(r_file_list)):
        if(len(r_file_list[_site]) != num_r_files_per_site):
            errors.append('Inconsistent number of response files at site {}'.format(_site))

    return errors


def _noerror_open(filename: str, file_handle=gdal.GA_ReadOnly) -> gdal.Dataset:
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    dataset = gdal.Open(filename, file_handle)
    gdal.PopErrorHandler()
    return dataset
