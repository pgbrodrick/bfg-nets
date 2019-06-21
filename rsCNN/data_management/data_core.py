
from typing import List, Tuple
import logging
import gdal
import numpy as np
import os

from rsCNN.configuration import configs, sections
from rsCNN.data_management import scalers, training_data
from rsCNN.data_management.sequences import MemmappedSequence


_FILENAME_BUILT_DATA_CONFIG_SUFFIX = 'built_data_config.yaml'
_FILENAME_FEATURES_SUFFIX = 'features_{}.npy'
_FILENAME_FEATURES_TEMPORARY_SUFFIX = '_features_memmap_temporary.npy'
_FILENAME_RESPONSES_SUFFIX = 'responses_{}.npy'
_FILENAME_RESPONSES_TEMPORARY_SUFFIX = '_responses_memmap_temporary.npy'
_FILENAME_WEIGHTS_SUFFIX = 'weights_{}.npy'
_FILENAME_WEIGHTS_TEMPORARY_SUFFIX = '_weights_memmap_temporary.npy'
_FILENAME_DATA_CONTAINER = 'data_container.npz'
_VECTORIZED_FILENAMES = ('kml', 'shp')


_logger = logging.getLogger(__name__)


class DataContainer:
    """ A container class that holds all sorts of data objects
    """
    config = None
    features = list()
    responses = list()
    weights = list()
    training_sequence = None
    validation_sequence = None

    feature_band_types = None
    response_band_types = None
    feature_raw_band_types = None
    response_raw_band_types = None

    feature_scaler = list()
    response_scaler = list()

    train_folds = None

    def __init__(self, config: configs.Config):
        errors = config.get_human_readable_config_errors(include_sections=['raw_files', 'data_build', 'data_samples'])
        assert not errors, errors
        self.config = config

        if (os.path.isfile(get_built_data_container_filepath(self.config))):
            _logger.info('Previously saved DataContainer found')
            try:
                self._load_data_core()
            except:
                _logger.info('Failed to load previously saved DataContainer')

    def build_or_load_rawfile_data(self, rebuild: bool = False):

        # Load data if it already exists
        if training_data.check_built_data_files_exist(self.config) and not rebuild:
            features, responses, weights = training_data.load_built_data_files(self.config)

        else:
            errors = sections.check_input_file_validity(self.config.raw_files.feature_files,
                                                        self.config.raw_files.response_files,
                                                        self.config.raw_files.boundary_files)

            create_built_data_output_directory(self.config)

            if (self.config.raw_files.ignore_projections is False):
                errors.extend(training_data.check_projections(self.config.raw_files.feature_files,
                                                              self.config.raw_files.response_files,
                                                              self.config.raw_files.boundary_files))

            if (self.config.raw_files.boundary_files is not None):
                boundary_files = [[loc_file for loc_file in self.config.raw_files.boundary_files
                                   if gdal.Open(loc_file, gdal.GA_ReadOnly) is not None]]
            else:
                boundary_files = None

            errors.extend(training_data.check_resolutions(self.config.raw_files.feature_files,
                                                          self.config.raw_files.response_files, boundary_files))

            errors.extend(self.check_band_types(self.config.raw_files.feature_files,
                                                self.config.raw_files.feature_data_type))
            errors.extend(self.check_band_types(self.config.raw_files.response_files,
                                                self.config.raw_files.response_data_type))

            if (len(errors) > 0):
                print('List of raw data file format errors is as follows:\n' + '\n'.join(error for error in errors))
            assert not errors, 'Raw data file errors found, terminating'

            self.feature_raw_band_types = self.get_band_types(
                self.config.raw_files.feature_files, self.config.raw_files.feature_data_type)
            self.response_raw_band_types = self.get_band_types(
                self.config.raw_files.response_files, self.config.raw_files.response_data_type)

            if (self.config.data_build.network_category == 'FCN'):
                features, responses, weights, feature_band_types, response_band_types = training_data.build_training_data_ordered(
                    self.config, self.feature_raw_band_types, self.response_raw_band_types)
            elif (self.config.data_build.network_category == 'CNN'):
                features, responses, weights, feature_band_types, response_band_types = \
                    training_data.build_training_data_from_response_points(
                        self.config, self.feature_raw_band_types, self.response_raw_band_types)
            else:
                raise NotImplementedError('Unknown response data format')

            self.feature_band_types = feature_band_types
            self.response_band_types = response_band_types

            _logger.info('Saving DataContainer')
            self._save_data_core()

        self.features = features
        self.responses = responses
        self.weights = weights

    def build_or_load_scalers(self, rebuild=False):

        # TODO:  I think this worked only if feature_scaler_name was a string, but it was also possible to be a list
        #  according to the DataConfig, in which case it would error out. This needs to be updated for multiple scalers.
        #  Specifically, the feature_scaler and response_scaler assignments need to be vectorized.
        basename = get_memmap_basename(self.config)
        feat_scaler_atr = {'savename_base': basename + '_feature_scaler'}
        feature_scaler = scalers.get_scaler(self.config.data_samples.feature_scaler_names[0], feat_scaler_atr)
        resp_scaler_atr = {'savename_base': basename + '_response_scaler'}
        response_scaler = scalers.get_scaler(self.config.data_samples.response_scaler_names[0], resp_scaler_atr)
        feature_scaler.load()
        response_scaler.load()

        self.train_folds = [x for x in range(self.config.data_build.number_folds)
                            if x not in (self.config.data_build.validation_fold, self.config.data_build.test_fold)]

        if (feature_scaler.is_fitted is False or rebuild is True):
            # TODO: do better
            feature_scaler.fit(self.features[self.train_folds[0]])
            feature_scaler.save()
        if (response_scaler.is_fitted is False or rebuild is True):
            # TODO: do better
            response_scaler.fit(self.responses[self.train_folds[0]])
            response_scaler.save()

        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler

    def load_sequences(self):
        train_folds = [idx for idx in range(self.config.data_build.number_folds)
                       if idx not in (self.config.data_build.validation_fold, self.config.data_build.test_fold)]

        self.training_sequence = self._build_memmapped_sequence(train_folds)
        self.validation_sequence = self._build_memmapped_sequence([self.config.data_build.validation_fold])

    def _build_memmapped_sequence(self, fold_indices: List[int]) -> MemmappedSequence:
        errors = []
        if (self.features is None):
            errors.append('data_container must have loaded feature numpy files')
        if(self.responses is None):
            errors.append('data_container must have loaded responses numpy files')
        if(self.weights is None):
            errors.append('data_container must have loaded weight numpy files')

        if(self.feature_scaler is None):
            errors.append('Feature scaler must be defined')
        if(self.response_scaler is None):
            errors.append('Response scaler must be defined')

        if(self.config.data_samples.batch_size is None):
            errors.append('config.data_samples.batch_size must be defined')

        if (len(errors) > 0):
            print('List of memmap sequence errors is as follows:\n' + '\n'.join(error for error in errors))
        assert not errors, 'Memmap sequence build errors found, terminating'

        data_sequence = MemmappedSequence(
            [self.features[_f] for _f in fold_indices],
            [self.responses[_r] for _r in fold_indices],
            [self.weights[_w] for _w in fold_indices],
            self.feature_scaler,
            self.response_scaler,
            self.config.data_samples.batch_size,
            apply_random_transforms=self.config.data_samples.apply_random_transformations,
            feature_mean_centering=self.config.data_build.feature_mean_centering,
            nan_replacement_value=self.config.data_samples.feature_nodata_encoding
        )
        return data_sequence

    def check_band_types(self, file_list, band_types):
        errors = []
        valid_band_types = ['R', 'C']
        # 3 options are available for specifying band_types:
        # 1) band_types is None - assume all bands are real
        # 2) band_types is a list of strings within valid_band_types - assume each band from the associated file is the
        #    specified type, requires len(band_types) == len(file_list[0])
        # 3) band_types is list of lists (of strings, contained in valid_band_types), with the outer list referring to
        #    files and the inner list referring to bands

        num_bands_per_file = [gdal.Open(x, gdal.GA_ReadOnly).RasterCount for x in file_list[0]]

        if (band_types is not None):

            if (type(band_types) is not list):
                errors.append('band_types must be None or a list')
                if (len(errors) > 0):
                    errors.append('All band type checks could not be completed')
                    return errors

            # List of lists, option 3 from above - just check components
            if (type(band_types[0]) is list):
                for _file in range(len(band_types)):
                    if (type(band_types[_file]) is not list):
                        errors.append('If one element of band_types is a list, all elements must be lists')
                    if (len(band_types[_file]) != num_bands_per_file[_file]):
                        errors.append('File {} has wrong number of band types'.format(_file))
                    for _band in range(len(band_types[_file])):
                        if (band_types[_file][_band] not in valid_band_types):
                            errors.append('Invalid band types at file {}, band {}'.format(_file, _band))

            else:
                if (len(band_types) != len(file_list)):
                    errors.append('Length of band_types is not equal to length of file list.  Incorrect input format')
                for _file in range(len(band_types)):
                    if (band_types[_file] not in valid_band_types):
                        errors.append('Invalid band type at File {}'.format(_file))

        return errors

    def get_band_types(self, file_list, band_types):
        valid_band_types = ['R', 'C']
        # 3 options are available for specifying band_types:
        # 1) band_types is None - assume all bands are real
        # 2) band_types is a list of strings within valid_band_types - assume each band from the associated file is the
        #    specified type, requires len(band_types) == len(file_list[0])
        # 3) band_types is list of lists (of strings, contained in valid_band_types), with the outer list referring to
        #    files and the inner list referring to bands

        num_bands_per_file = [gdal.Open(x, gdal.GA_ReadOnly).RasterCount for x in file_list[0]]

        # Nonetype, option 1 from above, auto-generate
        if (band_types is None):
            for _file in range(len(file_list[0])):
                output_raw_band_types = list()
                output_raw_band_types.append(['R' for _band in range(num_bands_per_file[_file])])

        else:

            if (type(band_types[0]) is list):
                output_raw_band_types = band_types
            else:
                # List of values valid_band_types, option 2 from above - convert to list of lists
                output_raw_band_types = []
                for _file in range(len(band_types)):
                    output_raw_band_types.append([band_types[_file] for _band in range(num_bands_per_file[_file])])

        # since it's more convenient, flatten this list of lists into a list before returning
        output_raw_band_types = [item for sublist in output_raw_band_types for item in sublist]

        return output_raw_band_types

    def _save_data_core(self):
        np.savez(get_built_data_container_filepath(self.config),
                 feature_band_types=self.feature_band_types,
                 response_band_types=self.response_band_types,
                 feature_raw_band_types=self.feature_raw_band_types,
                 response_raw_band_types=self.response_raw_band_types,
                 train_folds=self.train_folds
                 )

    def _load_data_core(self):
        npzf = np.load(get_built_data_container_filepath(self.config))
        self.feature_band_types = npzf['feature_band_types']
        self.response_band_types = npzf['response_band_types']
        self.feature_raw_band_types = npzf['feature_raw_band_types']
        self.response_raw_band_types = npzf['response_raw_band_types']
        self.train_folds = npzf['train_folds']


################### Filepath Nomenclature Functions ##############################


def create_built_data_output_directory(config: configs.Config) -> None:
    if not os.path.exists(config.data_build.dir_out):
        _logger.debug('Create built data output directory at {}'.format(config.data_build.dir_out))
        os.makedirs(config.data_build.dir_out)


def get_temporary_features_filepath(config: configs.Config) -> str:
    return get_temporary_data_filepaths(config, _FILENAME_FEATURES_TEMPORARY_SUFFIX)


def get_temporary_responses_filepath(config: configs.Config) -> str:
    return get_temporary_data_filepaths(config, _FILENAME_RESPONSES_TEMPORARY_SUFFIX)


def get_temporary_weights_filepath(config: configs.Config) -> str:
    return get_temporary_data_filepaths(config, _FILENAME_WEIGHTS_TEMPORARY_SUFFIX)


def get_temporary_data_filepaths(config: configs.Config, filename_suffix: str) -> str:
    return get_memmap_basename(config) + filename_suffix


def get_built_features_filepaths(config: configs.Config) -> List[str]:
    return get_built_data_filepaths(config, _FILENAME_FEATURES_SUFFIX)


def get_built_responses_filepaths(config: configs.Config) -> List[str]:
    return get_built_data_filepaths(config, _FILENAME_RESPONSES_SUFFIX)


def get_built_weights_filepaths(config: configs.Config) -> List[str]:
    return get_built_data_filepaths(config, _FILENAME_WEIGHTS_SUFFIX)


def get_built_data_config_filepath(config: configs.Config) -> str:
    return get_built_data_filepaths(config, _FILENAME_BUILT_DATA_CONFIG_SUFFIX)[0]


def get_built_data_filepaths(config: configs.Config, filename_suffix: str) -> List[str]:
    basename = get_memmap_basename(config)
    filepaths = [basename + filename_suffix.format(idx_fold) for idx_fold in range(config.data_build.number_folds)]
    return filepaths


def get_memmap_basename(config: configs.Config) -> str:
    filepath_separator = config.data_build.filename_prefix_out + '_' if config.data_build.filename_prefix_out else ''
    return os.path.join(config.data_build.dir_out, filepath_separator)


def get_built_data_container_filepath(config: configs.Config) -> str:
    return os.path.join(config.data_build.dir_out, _FILENAME_DATA_CONTAINER)
