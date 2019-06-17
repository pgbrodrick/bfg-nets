
from typing import List, Tuple
import logging
import gdal
import os

from rsCNN.configuration import configs
from rsCNN.data_management import scalers


_FILENAME_BUILT_DATA_CONFIG_SUFFIX = 'built_data_config.yaml'
_FILENAME_FEATURES_SUFFIX = 'features_{}.npy'
_FILENAME_FEATURES_TEMPORARY_SUFFIX = '_features_memmap_temporary.npy'
_FILENAME_RESPONSES_SUFFIX = 'responses_{}.npy'
_FILENAME_RESPONSES_TEMPORARY_SUFFIX = '_responses_memmap_temporary.npy'
_FILENAME_WEIGHTS_SUFFIX = 'weights_{}.npy'
_FILENAME_WEIGHTS_TEMPORARY_SUFFIX = '_weights_memmap_temporary.npy'
_VECTORIZED_FILENAMES = ('kml', 'shp')


_logger = logging.getLogger(__name__)


class Data_Container:
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

    # TODO:  not used!
    feature_scalers = list()
    response_scalers = list()

    train_folds = None

    def __init__(self, config: configs.Config):
        self.config = config

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

    def check_input_files(self, f_file_list, r_file_list, b_file_list):

        # f = feature, r = response, b = boundary

        # file lists r and f are expected a list of lists.  The outer list is a series of sites (location a, b, etc.).
        # The inner list is a series of files associated with that site (band x, y, z).  Each site must have the
        # same number of files, and each file from each site must have the same number of bands, in the same order.
        # file list b is a list for each site, with one boundary file expected to be the interior boundary for all
        # bands.
        # TODO:  move these checks to configs.py
        # Check that feature and response files are lists
        assert type(f_file_list) is list, 'Feature files must be a list of lists'
        assert type(r_file_list) is list, 'Response files must be a list of lists'

        # Checks on the matching numbers of sites
        assert len(f_file_list) == len(r_file_list), 'Feature and response site lists must be the same length'
        assert len(f_file_list) > 0, 'At least one feature and response site is required'
        if b_file_list is not None:
            assert len(b_file_list) == len(f_file_list), \
                'Boundary and feature file lists must be the same length. Boundary list: {}. Feature list: {}.'.format(
                    b_file_list, f_file_list)

        # Checks that we have lists of lists for f and r
        for _f in range(len(f_file_list)):
            assert type(f_file_list[_f]) is list, 'Features at site {} are not as a list'.format(_f)
            assert type(r_file_list[_f]) is list, 'Responses at site {} are not as a list'.format(_f)

        # Checks that all files can be opened by gdal
        for _site in range(len(f_file_list)):
            assert type(f_file_list[_site]) is list, 'Features at site {} are not as a list'.format(_site)
            assert type(r_file_list[_site]) is list, 'Responses at site {} are not as a list'.format(_site)
            for _band in range(len(f_file_list[_site])):
                assert gdal.Open(f_file_list[_site][_band], gdal.GA_ReadOnly) is not None,\
                    'Could not open feature site {}, file {}'.format(_site, _band)
            for _band in range(len(r_file_list[_site])):
                assert gdal.Open(r_file_list[_site][_band], gdal.GA_ReadOnly) is not None,\
                    'Could not open response site {}, file {}'.format(_site, _band)

        # Checks on the number of files per site
        num_f_files_per_site = len(f_file_list[0])
        num_r_files_per_site = len(r_file_list[0])
        for _site in range(len(f_file_list)):
            assert len(f_file_list[_site]) == num_f_files_per_site, \
                'Inconsistent number of feature files at site {}'.format(_site)
            assert len(r_file_list[_site]) == num_r_files_per_site, \
                'Inconsistent number of response files at site {}'.format(_site)

        # Checks on the number of bands per file
        num_f_bands_per_file = [gdal.Open(x, gdal.GA_ReadOnly).RasterCount for x in f_file_list[0]]
        num_r_bands_per_file = [gdal.Open(x, gdal.GA_ReadOnly).RasterCount for x in r_file_list[0]]
        for _site in range(len(f_file_list)):
            for _file in range(len(f_file_list[_site])):
                assert gdal.Open(f_file_list[_site][_file], gdal.GA_ReadOnly).RasterCount == num_f_bands_per_file[_file],\
                    'Inconsistent number of feature bands in site {}, file {}'.format(_site, _band)

            for _file in range(len(r_file_list[_site])):
                assert gdal.Open(r_file_list[_site][_file], gdal.GA_ReadOnly).RasterCount == num_r_bands_per_file[_file],\
                    'Inconsistent number of response bands in site {}, file {}'.format(_site, _band)

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
            assert type(band_types) is list, 'band_types must be None or a list'

            # List of lists, option 3 from above - just check components
            if (type(band_types[0]) is list):
                for _file in range(len(band_types)):
                    assert type(band_types[_file]) is list, \
                        'If one element of band_types is a list, all elements must be lists'
                    assert len(band_types[_file]) == num_bands_per_file[_file], \
                        'File {} has wrong number of band types'.format(_file)
                    for _band in range(len(band_types[_file])):
                        assert band_types[_file][_band] in valid_band_types, \
                            'Invalid band types at file {}, band {}'.format(_file, _band)

                output_raw_band_types = band_types

            else:
                # List of values valid_band_types, option 2 from above - convert to list of lists
                output_raw_band_types = []
                for _file in range(len(band_types)):
                    assert band_types[_file] in valid_band_types, 'Invalid band type at File {}'.format(_file)
                    output_raw_band_types.append([band_types[_file] for _band in range(num_bands_per_file[_file])])

        # since it's more convenient, flatten this list of lists into a list before returning
        output_raw_band_types = [item for sublist in output_raw_band_types for item in sublist]

        return output_raw_band_types



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


