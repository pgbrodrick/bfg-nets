import logging
import os
from pathlib import Path
import re
from typing import List, Tuple

import fiona
import gdal
import numpy as np
import numpy.matlib
import ogr
import rasterio.features
from tqdm import tqdm

from rsCNN.configuration import configs
from rsCNN.data_management import scalers
from rsCNN.data_management import common_io
# TODO:  remove * imports
from rsCNN.utils.general import *


_logger = logging.getLogger(__name__)


# TODO: change munge to temporary
_FILENAME_BUILT_DATA_CONFIG_SUFFIX = 'built_data_config.yaml'
_FILENAME_FEATURES_SUFFIX = 'features_{}.npy'
_FILENAME_FEATURES_TEMPORARY_SUFFIX = '_features_memmap_temporary.npy'
_FILENAME_RESPONSES_SUFFIX = 'responses_{}.npy'
_FILENAME_RESPONSES_TEMPORARY_SUFFIX = '_responses_memmap_temporary.npy'
_FILENAME_WEIGHTS_SUFFIX = 'weights_{}.npy'
_FILENAME_WEIGHTS_TEMPORARY_SUFFIX = '_weights_memmap_temporary.npy'
_VECTORIZED_FILENAMES = ('kml', 'shp')


def rasterize_vector(vector_file, geotransform, output_shape):
    """ Rasterizes an input vector directly into a numpy array.
    Arguments:
    vector_file - str
      Input vector file to be rasterized.
    geotransform - list
      A gdal style geotransform.
    output_shape - tuple
      The shape of the output file to be generated.

    Return:
    A rasterized 2-d numpy array.
    """
    ds = fiona.open(vector_file, 'r')
    geotransform = [geotransform[1], geotransform[2], geotransform[0],
                    geotransform[4], geotransform[5], geotransform[3]]
    mask = np.zeros(output_shape)
    for n in range(0, len(ds)):
        rasterio.features.rasterize([ds[n]['geometry']], transform=geotransform, default_value=1, out=mask)
    return mask


def build_or_load_rawfile_data(config: configs.Config, rebuild: bool = False):
    data_container = Dataset(config)
    data_container.check_input_files(
        config.raw_files.feature_files, config.raw_files.response_files,
        config.raw_files.boundary_files
    )

    data_container.feature_raw_band_types = data_container.get_band_types(
        config.raw_files.feature_files, config.raw_files.feature_data_type)
    data_container.response_raw_band_types = data_container.get_band_types(
        config.raw_files.response_files, config.raw_files.response_data_type)

    # Load data if it already exists
    if _check_built_data_files_exist(config) and not rebuild:
        features, responses, weights = _load_built_data_files(config)

    else:
        assert config.raw_files.feature_files is not [], 'feature files to pull data from are required'
        assert config.raw_files.response_files is not [], 'response files to pull data from are required'
        _create_built_data_output_directory(config)

        if (config.raw_files.ignore_projections is False):
            check_projections(config.raw_files.feature_files,
                              config.raw_files.response_files,
                              config.raw_files.boundary_files
                              )

        if (config.raw_files.boundary_files is not None):
            boundary_files = [[loc_file for loc_file in config.raw_files.boundary_files
                               if gdal.Open(loc_file, gdal.GA_ReadOnly) is not None]]
        else:
            boundary_files = None

        check_resolutions(config.raw_files.feature_files, config.raw_files.response_files, boundary_files)

        if (config.data_build.response_data_format == 'FCN'):
            features, responses, weights, feature_band_types, response_band_types = build_training_data_ordered(
                config, data_container.feature_raw_band_types, data_container.response_raw_band_types)
        elif (config.data_build.response_data_format == 'CNN'):
            features, responses, weights, feature_band_types, response_band_types = \
                build_training_data_from_response_points(
                    config, data_container.feature_raw_band_types, data_container.response_raw_band_types)
        else:
            raise NotImplementedError('Unknown response data format')

        data_container.feature_band_types = feature_band_types
        data_container.response_band_types = response_band_types
    """
    TODO:  Phil:  we shouldn't be assigning attributes to data_container, data_container should assign attributes to 
    itself, and this will also simplify the API. Currently:
    
    # Create dataset
    data_container = training_data.build_or_load_rawfile_data(config, rebuild=False)
    data_container.build_or_load_scalers()
    # Create sequences
    train_folds = [idx for idx in range(config.data_build.number_folds)
                   if idx not in (config.data_build.validation_fold, config.data_build.test_fold)]
    training_sequence = sequences.build_memmapped_sequence(
        data_container, train_folds, batch_size=config.data_samples.batch_size)
    validation_sequence = sequences.build_memmapped_sequence(
        data_container, [config.data_build.validation_fold], batch_size=config.data_samples.batch_size)
        
    Instead:
    
    data_container = DataContainer(config)
    data_container.build_formatted_data_files() <-- takes a while, so should be explicit, should pass if already built
    data_container.load_data_sequences() <-- should handle scalers and all sequences being created
    
    Are memmapped loads free? If so, then don't save features/responses/weights from build_formatted step. If not,
    then save to the data_container, but don't do anything with them.
    
    Also, can we be consistent about the name of dataset or data_container? Use one for the class and instance names?
    """
    data_container.features = features
    data_container.responses = responses
    data_container.weights = weights

    return data_container


def get_proj(fname):
    """ Get the projection of a raster/vector dataset.
    Arguments:
    fname - str
      Name of input file.
    is_vector - boolean
      Boolean indication of whether the file is a vector or a raster.

    Returns:
    The projection of the input fname
    """
    ds = gdal.Open(fname, gdal.GA_ReadOnly)
    if (ds is not None):
        b_proj = ds.GetProjection()
        if (b_proj is not None):
            return b_proj

    if (os.path.basename(fname).split('.')[-1] == 'shp'):
        vset = ogr.GetDriverByName('ESRI Shapefile').Open(fname, gdal.GA_ReadOnly)
    elif (os.path.basename(fname).split('.')[-1] == 'kml'):
        vset = ogr.GetDriverByName('KML').Open(fname, gdal.GA_ReadOnly)
    else:
        raise Exception('Cannot find projection from file {}'.format(fname))

    if (vset is None):
        raise Exception('Cannot find projection from file {}'.format(fname))
    else:
        b_proj = vset.GetLayer().GetSpatialRef()
        if (b_proj is None):
            raise Exception('Cannot find projection from file {}'.format(fname))
        else:
            b_proj = re.sub('\W', '', str(b_proj))

    return b_proj


def check_projections(a_files, b_files, c_files=None):

    loc_a_files = [item for sublist in a_files for item in sublist]
    loc_b_files = [item for sublist in b_files for item in sublist]
    if c_files is None:
        loc_c_files = list()
    else:
        loc_c_files = [item for sublist in c_files for item in sublist]

    a_proj = []
    b_proj = []
    c_proj = []

    for _f in range(len(loc_a_files)):
        a_proj.append(get_proj(loc_a_files[_f]))
        b_proj.append(get_proj(loc_b_files[_f]))
        if (len(loc_c_files) > 0):
            c_proj.append(get_proj(loc_c_files[_f]))

    for _p in range(len(a_proj)):
        if (len(c_proj) > 0):
            assert (a_proj[_p] == b_proj[_p] and a_proj == c_proj[_p]), \
                'Projection_mismatch between\n{}: {},\n{}: {},\n{}: {}'.format(
                    a_proj[_p], loc_a_files[_p], b_proj[_p], loc_b_files[_p], c_proj[_p], loc_c_files[_p])
        else:
            assert (a_proj[_p] == b_proj[_p]), 'Projection_mismatch between\n{}: {}\n{}: {}'.\
                format(a_proj[_p], loc_a_files[_p], b_proj[_p], loc_b_files[_p])


def check_resolutions(a_files, b_files, c_files=None):
    if c_files is None:
        c_files = list()

    loc_a_files = [item for sublist in a_files for item in sublist]
    loc_b_files = [item for sublist in b_files for item in sublist]
    if (len(c_files) > 0):
        loc_c_files = [item for sublist in c_files for item in sublist]
    else:
        loc_c_files = []

    a_res = []
    b_res = []
    c_res = []

    for _f in range(len(loc_a_files)):
        a_res.append(np.array(gdal.Open(loc_a_files[_f], gdal.GA_ReadOnly).GetGeoTransform())[[1, 5]])
        b_res.append(np.array(gdal.Open(loc_b_files[_f], gdal.GA_ReadOnly).GetGeoTransform())[[1, 5]])
        if (len(loc_c_files) > 0):
            c_res.append(np.array(gdal.Open(loc_c_files[_f], gdal.GA_ReadOnly).GetGeoTransform())[[1, 5]])

    for _p in range(len(a_res)):
        if (len(c_res) > 0):
            assert (np.all(a_res[_p] == b_res[_p]) and np.all(a_res == c_res[_p])), \
                'Resolution mismatch between\n{}: {},\n{}: {},\n{}: {}'.format(
                    a_res[_p], loc_a_files[_p], b_res[_p], loc_b_files[_p], c_res[_p], loc_c_files[_p])
        else:
            assert (np.all(a_res[_p] == b_res[_p])), 'Resolution mimatch between\n{}: {}\n{}: {}'.\
                format(a_res[_p], loc_a_files[_p], b_res[_p], loc_b_files[_p])

# Calculates categorical weights for a single response


def calculate_categorical_weights(
        responses: List[np.array],
        weights: List[np.array],
        config: configs.Config,
        batch_size: int = 100
) -> List[np.array]:

    # find upper and lower boud
    lb = config.data_build.window_radius - config.data_build.loss_window_radius
    ub = -lb

    # get response/total counts (batch-wise)
    response_counts = np.zeros(responses[0].shape[-1])
    total_valid_count = 0
    for idx_array, response_array in enumerate(responses):
        if idx_array in (config.data_build.validation_fold, config.data_build.test_fold):
            continue
        for ind in range(0, response_array.shape[0], batch_size):
            if (lb == 0):
                lr = response_array[ind:ind+batch_size, ...]
            else:
                lr = response_array[ind:ind+batch_size, lb:ub, lb:ub, :]
            lr[lr == config.raw_files.response_nodata_value] = np.nan
            total_valid_count += np.sum(np.isfinite(lr))
            for _r in range(0, len(response_counts)):
                response_counts[_r] += np.nansum(lr[..., _r] == 1)

    # assign_weights
    for _array in range(len(responses)):
        for ind in range(0, responses[_array].shape[0], batch_size):

            lr = (responses[_array])[ind:ind+batch_size, ...]
            lrs = list(lr.shape)
            lrs.pop(-1)
            lw = np.zeros((lrs))
            for _r in range(0, len(response_counts)):
                lw[lr[..., _r] == 1] = total_valid_count / response_counts[_r]

            if (lb != 0):
                lw[:, :lb, :] = 0
                lw[:, ub:, :] = 0
                lw[:, :, :lb] = 0
                lw[:, :, ub:] = 0

            lws = list(lw.shape)
            lws.extend([1])
            lw = lw.reshape(lws)
            weights[_array][ind:ind+batch_size, ...] = lw

    return weights


def read_mask_chunk(
        boundary_vector_file: str,
        boundary_subset_geotransform: tuple,
        b_set: gdal.Dataset,
        boundary_upper_left: List[int],
        window_diameter: int,
        boundary_bad_value: float
) -> np.array:
    # Start by checking if we're inside boundary, if there is one
    mask = None
    if (boundary_vector_file is not None):
        mask = rasterize_vector(boundary_vector_file, boundary_subset_geotransform,
                                (window_diameter, window_diameter))
    if (b_set is not None):
        mask = b_set.ReadAsArray(int(boundary_upper_left[0]), int(
            boundary_upper_left[1]), window_diameter, window_diameter)

    if mask is None:
        mask = np.zeros((window_diameter, window_diameter)).astype(bool)
    else:
        mask = mask == boundary_bad_value

    return mask


def read_labeling_chunk(f_sets: List[gdal.Dataset],
                        feature_upper_lefts: List[List[int]],
                        config: configs.Config,
                        boundary_vector_file: str = None,
                        boundary_subset_geotransform: tuple = None,
                        b_set=None,
                        boundary_upper_left: List[int] = None):

    for _f in range(len(feature_upper_lefts)):
        if (np.any(feature_upper_lefts[_f] < config.data_build.window_radius)):
            _logger.debug('Feature read OOB')
            return None
        if (feature_upper_lefts[_f][0] > f_sets[_f].RasterXSize - config.data_build.window_radius):
            _logger.debug('Feature read OOB')
            return None
        if (feature_upper_lefts[_f][1] > f_sets[_f].RasterYSize - config.data_build.window_radius):
            _logger.debug('Feature read OOB')
            return None

    window_diameter = config.data_build.window_radius * 2

    mask = read_mask_chunk(boundary_vector_file,
                           boundary_subset_geotransform,
                           b_set,
                           boundary_upper_left,
                           window_diameter,
                           config.raw_files.boundary_bad_value)

    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        _logger.debug('Insufficient mask data')
        return None

    local_feature, mask = common_io.read_map_subset(f_sets, feature_upper_lefts,
                                                    window_diameter, mask, config.raw_files.feature_nodata_value)

    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        _logger.debug('Insufficient feature data')
        return None

    # Final check (propogate mask forward), and return
    local_feature[mask, :] = np.nan

    return local_feature


def read_segmentation_chunk(f_sets: List[tuple],
                            r_sets: List[tuple],
                            feature_upper_lefts: List[List[int]],
                            response_upper_lefts: List[List[int]],
                            config: configs.Config,
                            boundary_vector_file: str = None,
                            boundary_subset_geotransform: tuple = None,
                            b_set=None,
                            boundary_upper_left: List[int] = None):
    window_diameter = config.data_build.window_radius * 2

    mask = read_mask_chunk(boundary_vector_file,
                           boundary_subset_geotransform,
                           b_set,
                           boundary_upper_left,
                           window_diameter,
                           config.raw_files.boundary_bad_value)

    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        return None, None

    local_response, mask = common_io.read_map_subset(r_sets, response_upper_lefts,
                                                     window_diameter, mask, config.raw_files.response_nodata_value)
    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        return None, None

    if (config.data_build.response_min_value is not None):
        local_response[local_response < config.data_build.response_min_value] = np.nan
    if (config.data_build.response_max_value is not None):
        local_response[local_response > config.data_build.response_max_value] = np.nan
    mask[np.any(np.isnan(local_response), axis=-1)] = True

    if (mask is None):
        return None, None
    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        return None, None

    local_feature, mask = common_io.read_map_subset(f_sets, feature_upper_lefts,
                                                    window_diameter, mask, config.raw_files.feature_nodata_value)

    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        return None, None

    # Final check (propogate mask forward), and return
    local_feature[mask, :] = np.nan
    local_response[mask, :] = np.nan

    return local_feature, local_response


class Dataset:
    """ A container class that holds all sorts of data objects
    """
    # TODO:  Phil:  note that I moved these attribute definitions to the class itself, not the init, so that we can
    #  a) document them more easily and b) so that they're part of the class definition for IDE / other introspection.
    #  Please delete this after you see it.
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
        basename = _get_memmap_basename(self.config)
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


# TODO:  typing
def build_training_data_ordered(
        config: configs.Config,
        feature_raw_band_types: List[List[str]],
        response_raw_band_types: List[List[str]]
):

    # TODO:  check default, and set it to a standard value
    if config.data_build.random_seed:
        _logger.debug('Setting random seed to {}'.format(config.data_build.random_seed))
        np.random.seed(config.data_build.random_seed)

    # TODO:  move to config checks
    if (isinstance(config.data_build.max_samples, list)):
        if (len(config.data_build.max_samples) != len(config.raw_files.feature_files)):
            raise Exception('max_samples must equal feature_files length, or be an integer.')
    # TODO:  move to checks
    # TODO:  fix max size issue, but force for now to prevent overly sized sets
    n_features = int(np.sum([len(feat_type) for feat_type in feature_raw_band_types]))
    n_responses = int(np.sum([len(resp_type) for resp_type in response_raw_band_types]))
    assert config.data_build.max_samples * (config.data_build.window_radius*2)**2 * \
        n_features / 1024.**3 < 10, 'max_samples too large'

    features, responses = _create_munged_features_responses_data_files(config, n_features, n_responses)
    _log_munged_data_information(features, responses)

    _logger.debug('Pre-compute all subset locations')
    all_site_upper_lefts = []
    all_site_xy_locations = []
    gdal_datasets = []
    reference_subset_geotransforms = []
    for _site in range(0, len(config.raw_files.feature_files)):
        feature_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly)
                        for loc_file in config.raw_files.feature_files[_site]]
        response_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_files.response_files[_site]]
        boundary_set = _get_boundary_sets_from_boundary_files(config)[_site]

        all_set_upper_lefts, xy_sample_locations = common_io.get_all_interior_extent_subset_pixel_locations(\
                                                  gdal_datasets = [feature_sets, response_sets, [bs for bs in [boundary_set] if bs is not None]],\
                                                  window_radius = config.data_build.window_radius,\
                                                  inner_window_radius = config.data_build.loss_window_radius,\
                                                  shuffle = True)

        all_site_upper_lefts.append(all_set_upper_lefts)
        all_site_xy_locations.append(xy_sample_locations)
        gdal_datasets.append([feature_sets, response_sets, boundary_set])


        ref_trans = feature_sets[0].GetGeoTransform()
        subset_geotransform = None
        if config.raw_files.boundary_files is not None:
            if config.raw_files.boundary_files[_site] is not None and \
                    _is_boundary_file_vectorized(config.raw_files.boundary_files[_site]):
                subset_geotransform = [ref_trans[0], ref_trans[1], 0, ref_trans[3], 0, ref_trans[5]]
        reference_subset_geotransforms.append(subset_geotransform)



    _logger.debug('Step through sites in order and grab one sample from each until max samples')
    progress_bar = tqdm(total=config.data_build.max_samples, ncols=80)

    _sample_index = 0
    _site = 0
    _site_xy_index = np.zeros(len(gdal_datasets)).astype(int).tolist()
    while (_sample_index < config.data_build.max_samples and len(gdal_datasets) > 0):

        [feature_sets, response_sets, boundary_set] = gdal_datasets[_site]

        while _site_xy_index[_site] < len(all_site_upper_lefts[_site]):

            [f_ul, r_ul, b_ul] = all_site_upper_lefts[_site]

            local_boundary_vector_file = None
            local_boundary_upper_left = None
            if (b_ul is not None):
                local_boundary_upper_left = b_ul + all_site_xy_locations[_site][_site_xy_index, :]
            subset_geotransform = None
            if (reference_subset_geotransforms[_site] is not None):
                ref_trans = reference_subset_geotransforms[_site]
                subset_geotransform[0] = ref_trans[0] + (f_ul[0][0] + all_site_xy_locations[_site_xy_index[_site], 0]) * ref_trans[1]
                subset_geotransform[3] = ref_trans[_site][3] + (f_ul[0][1] + all_site_xy_locations[_site_xy_index[_site], 1]) * ref_trans[5]
                local_boundary_vector_file = config.raw_files.boundary_files[_site]

            local_feature, local_response = read_segmentation_chunk(feature_sets,
                                                                    response_sets,
                                                                    f_ul + all_site_xy_locations[_site][_site_xy_index[_site], :],
                                                                    r_ul + all_site_xy_locations[_site][_site_xy_index[_site], :],
                                                                    config,
                                                                    boundary_vector_file=local_boundary_vector_file,
                                                                    boundary_upper_left=local_boundary_upper_left,
                                                                    b_set=boundary_set,
                                                                    boundary_subset_geotransform=subset_geotransform)
            _site_xy_index[_site] += 1
            popped = None
            if (_site_xy_index[_site] >= len(all_site_xy_locations[_site])):
                _logger.debug('All locations in site {} have been checked.'.format(config.raw_files.feature_files[_site][0]))

                _site_xy_index.pop(_site)
                all_site_xy_locations.pop(_site)
                all_site_upper_lefts.pop(_site)
                reference_subset_geotransforms.pop(_site)
                popped = gdal_datasets.pop(_site)


            if local_feature is not None:
                _logger.debug('Save sample data; {} samples saved'.format(_sample_index + 1))
                features[_sample_index, ...] = local_feature.copy()
                responses[_sample_index, ...] = local_response.copy()
                _sample_index += 1
                progress_bar.update(1)

                if (popped is not None):
                    _site += 1

                break

    progress_bar.close()


    features = _resize_munged_features(features, _sample_index, config)
    responses = _resize_munged_responses(responses, _sample_index, config)
    _log_munged_data_information(features, responses)

    _logger.debug('Shuffle data to avoid fold assignment biases')
    perm = np.random.permutation(features.shape[0])
    features = features[perm, :]
    responses = responses[perm, :]
    del perm

    _logger.debug('Create uniform weights')
    shape = tuple(list(features.shape)[:-1] + [1])
    weights = np.memmap(_get_temporary_weights_filepath(config), dtype=np.float32, mode='w+', shape=shape)
    weights[:, :, :, :] = 1
    _logger.debug('Remove weights for missing responses')
    weights[np.isnan(responses[..., 0])] = 0

    _logger.debug('Remove weights outside loss window')
    if (config.data_build.loss_window_radius != config.data_build.window_radius):
        buf = config.data_build.window_radius - config.data_build.loss_window_radius
        weights[:, :buf, :, -1] = 0
        weights[:, -buf:, :, -1] = 0
        weights[:, :, :buf, -1] = 0
        weights[:, :, -buf:, -1] = 0
    _log_munged_data_information(features, responses, weights)

    _logger.debug('One-hot encode features')
    features, feature_band_types = common_io.one_hot_encode_array(
        feature_raw_band_types, features, _get_temporary_features_filepath(config))
    _logger.debug('One-hot encode responses')
    responses, response_band_types = common_io.one_hot_encode_array(
        response_raw_band_types, responses, _get_temporary_responses_filepath(config))
    _log_munged_data_information(features, responses, weights)

    _save_built_data_files(features, responses, weights, config)
    del features, responses, weights

    if ('C' in response_raw_band_types):
        assert np.sum(np.array(response_raw_band_types) == 'C') == 1, \
            'Weighting is currently only enabled for one categorical response variable.'
        features, responses, weights = _load_built_data_files(config, writeable=True)
        weights = calculate_categorical_weights(responses, weights, config)
        _logger.debug('Delete in order to flush output')
        del features, responses, weights

    _remove_temporary_data_files(config)

    _logger.debug('Store data build config sections')
    _save_built_data_config_sections_to_verify_successful(config)
    features, responses, weights = _load_built_data_files(config, writeable=False)
    return features, responses, weights, feature_band_types, response_band_types


# TODO:  typing
def build_training_data_from_response_points(
        config: configs.Config,
        feature_raw_band_types: List[List[str]],
        response_raw_band_types: List[List[str]]
):
    _logger.info('Build training data from response points')
    if (config.data_build.random_seed is not None):
        np.random.seed(config.data_build.random_seed)

    num_features = np.sum([len(feat_type) for feat_type in feature_raw_band_types])
    assert config.data_build.max_samples * (2 * config.data_build.window_radius) ** 2 * num_features / 1024**3 < 10, \
        'Max samples requested exceeds temporary and arbitrary threshold, we need to handle this to support more'

    xy_sample_points_per_site = []
    responses_per_site = []
    for _site in range(0, len(config.raw_files.feature_files)):
        _logger.debug('Open feature and response datasets for site {}'.format(_site))
        feature_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_files.feature_files[_site]]
        response_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_files.response_files[_site]]
        boundary_set = _get_boundary_sets_from_boundary_files(config)[_site]

        _logger.debug('Calculate overlapping extent')
        [f_ul, r_ul, b_ul], x_px_size, y_px_size = common_io.get_overlapping_extent(
            [feature_sets, response_sets, [bs for bs in [boundary_set] if bs is not None]])

        x_sample_points = []
        y_sample_points = []
        band_responses = []
        _logger.debug('Run through first response')
        for _line in range(y_px_size):
            line_dat = np.squeeze(response_sets[_site].ReadAsArray(r_ul[0][0], r_ul[0][1] + _line, x_px_size, 1))
            if len(line_dat.shape) == 1:
                line_dat = line_dat.reshape(-1, 1)

            if config.data_build.response_background_value is not None:
                good_data = np.all(line_dat != config.data_build.response_background_value, axis=1)
            else:
                good_data = np.ones(line_dat.shape[0]).astype(bool)

            if (config.raw_files.response_nodata_value is not None):
                good_data[np.any(line_dat == config.raw_files.response_nodata_value, axis=1)] = False

            if (np.sum(good_data) > 0):
                line_x = np.arange(x_px_size)
                line_y = line_x.copy()
                line_y[:] = _line

                x_sample_points.extend(line_x[good_data].tolist())
                y_sample_points.extend(line_y[good_data].tolist())
                band_responses.append(line_dat[good_data, :])

        xy_sample_points = np.vstack([np.array(x_sample_points), np.array(y_sample_points)]).T
        responses_per_file = [np.vstack(band_responses).astype(np.float32)]
        _logger.debug('Found {} responses for site {}'.format(len(responses_per_file[0]), _site))

        _logger.debug('Grab the rest of the resopnses')
        for _file in range(1, len(response_sets)):
            band_responses = np.zeros(responses_per_file[0].shape[0])
            for _point in range(len(xy_sample_points)):
                band_responses[_point] = response_sets[_file].ReadAsArray(
                    r_ul[_file][0], r_ul[_file][1], 1, 1).astype(np.float32)
            responses_per_file.append(band_responses.copy())

        responses_per_file = np.hstack(responses_per_file)
        _logger.debug('All responses now obtained, response stack of shape: {}'.format(responses_per_file.shape))

        _logger.debug('Check responses 1 onward for nodata values')
        good_data = np.all(responses_per_file != config.data_build.response_background_value, axis=1)
        xy_sample_points = xy_sample_points[good_data, :]

        xy_sample_points_per_site.append(xy_sample_points)
        responses_per_site.append(np.vstack(band_responses))

    total_samples = sum([site_responses.shape[0] for site_responses in responses_per_site])
    _logger.debug('Found {} total samples across {} sites'.format(total_samples, len(responses_per_site)))

    assert config.data_build.max_samples > 0, 'need at least 1 valid sample...'

    if total_samples > config.data_build.max_samples:
        _logger.debug('Discard samples because the number of valid samples ({}) exceeds the max samples requested ({})'
                      .format(total_samples, config.data_build.max_samples))

        prop_samples_kept_per_site = config.data_build.max_samples / total_samples
        for _site in range(len(responses_per_site)):
            num_samples = len(responses_per_site[_site])
            num_samples_kept = int(prop_samples_kept_per_site * num_samples)

            idxs_kept = np.random.permutation(num_samples)[:num_samples_kept]
            responses_per_site[_site] = responses_per_site[_site][idxs_kept, :]

            xy_sample_points_per_site[_site] = xy_sample_points_per_site[_site][idxs_kept, :]
            _logger.debug('Site {} had {} valid samples, kept {} samples'.format(_site, num_samples, num_samples_kept))

        total_samples = sum([site_responses.shape[0] for site_responses in responses_per_site])
        _logger.debug('Kept {} total samples across {} sites after discarding'.format(
            total_samples, len(responses_per_site)))

    # TODO: fix max size issue, but force for now to prevent overly sized sets
    features = np.memmap(
        _get_temporary_features_filepath(config), dtype=np.float32, mode='w+',
        shape=(total_samples, 2*config.data_build.window_radius, 2*config.data_build.window_radius, num_features)
    )

    sample_index = 0
    for _site in range(0, len(config.raw_files.feature_files)):
        _logger.debug('Open feature and response datasets for site {}'.format(_site))
        feature_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly)
                        for loc_file in config.raw_files.feature_files[_site]]
        response_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_files.response_files[_site]]
        boundary_set = _get_boundary_sets_from_boundary_files(config)[_site]

        _logger.debug('Calculate interior rectangle location and extent')
        [f_ul, r_ul, b_ul], x_px_size, y_px_size = common_io.get_overlapping_extent(
            [feature_sets, response_sets, [bs for bs in [boundary_set] if bs is not None]])

        # xy_sample_locations is current the response centers, but we need to use the pixel ULs.  So subtract
        # out the corresponding feature radius
        xy_sample_locations = xy_sample_points_per_site[_site] - config.data_build.window_radius

        ref_trans = feature_sets[0].GetGeoTransform()
        subset_geotransform = None
        if config.raw_files.boundary_files is not None:
            if config.raw_files.boundary_files[_site] is not None and \
                    _is_boundary_file_vectorized(config.raw_files.boundary_files[_site]):
                subset_geotransform = [ref_trans[0], ref_trans[1], 0, ref_trans[3], 0, ref_trans[5]]

        good_response_data = np.zeros(responses_per_site[_site].shape[0]).astype(bool)
        # Now read in features
        for _cr in tqdm(range(len(xy_sample_locations)), ncols=80):

            # Determine local information about boundary file
            local_boundary_vector_file = None
            local_boundary_upper_left = None
            if (boundary_set is not None):
                local_boundary_upper_left = b_ul + xy_sample_locations[_cr, :]
            if (subset_geotransform is not None):
                # define a geotramsform for the subset we are going to take
                subset_geotransform[0] = ref_trans[0] + (f_ul[0][0] + xy_sample_locations[_cr, 0]) * ref_trans[1]
                subset_geotransform[3] = ref_trans[3] + (f_ul[0][1] + xy_sample_locations[_cr, 1]) * ref_trans[5]
                local_boundary_vector_file = config.raw_files.boundary_files[_site]

            local_feature = read_labeling_chunk(
                feature_sets, f_ul + xy_sample_locations[_cr, :], config, boundary_vector_file=local_boundary_vector_file,
                boundary_upper_left=local_boundary_upper_left, b_set=boundary_set,
                boundary_subset_geotransform=subset_geotransform
            )

            # Make sure that the feature space also has data - the fact that the response space had valid data is no
            # guarantee that the feature space does.
            if local_feature is not None:
                _logger.debug('Save sample data; {} samples saved total'.format(sample_index + 1))
                features[sample_index, ...] = local_feature.copy()
                good_response_data[_cr] = True
                sample_index += 1
        responses_per_site[_site] = responses_per_site[_site][good_response_data, :]
        _logger.debug('{} samples saved for site {}'.format(responses_per_site[_site].shape[0], _site))

    assert sample_index > 0, 'Insufficient feature data corresponding to response data.  Consider increasing maximum feature nodata size'

    # transform responses
    responses = np.vstack(responses_per_site)
    del responses_per_site
    _log_munged_data_information(features, responses)

    features = _resize_munged_features(features, sample_index, config)
    _log_munged_data_information(features, responses)

    _logger.debug('Shuffle data to avoid fold assignment biases')
    perm = np.random.permutation(features.shape[0])
    features = features[perm, :]
    responses = responses[perm, :]
    del perm

    weights = np.ones((responses.shape[0], 1))
    _log_munged_data_information(features, responses, weights)

    # one hot encode
    features, feature_band_types = common_io.one_hot_encode_array(
        feature_raw_band_types, features, _get_temporary_features_filepath(config))
    responses, response_band_types = common_io.one_hot_encode_array(
        response_raw_band_types, responses, _get_temporary_responses_filepath(config))
    _log_munged_data_information(features, responses, weights)

    # This change happens to work in this instance, but will not work with all sampling strategies.  I will leave for
    # now and refactor down the line as necessary.  Generally speaking, fold assignments are specific to the style of data read
    _save_built_data_files(features, responses, weights, config)
    del features, responses, weights

    if 'C' in response_raw_band_types:
        assert np.sum(np.array(response_raw_band_types) == 'C') == 1, \
            'Weighting is currently only enabled for one categorical response variable.'
        features, responses, weights = _load_built_data_files(config, writeable=True)
        weights = calculate_categorical_weights(responses, weights, config)
        del features, responses, weights

    _remove_temporary_data_files(config)

    _save_built_data_config_sections_to_verify_successful(config)
    features, responses, weights = _load_built_data_files(config, writeable=False)
    return features, responses, weights, feature_band_types, response_band_types


################### Verification Functions ##############################

def _check_mask_data_sufficient(mask: np.array, max_nodata_fraction: float) -> bool:
    if mask is not None:
        nodata_fraction = np.sum(mask) / np.prod(mask.shape)
        if nodata_fraction <= max_nodata_fraction:
            _logger.debug('Data mask has sufficient data, missing data proportion: {}'.format(nodata_fraction))
            return True
        else:
            _logger.debug('Data mask has insufficient data, missing data proportion: {}'.format(nodata_fraction))
            return False
    else:
        _logger.debug('Data mask is None')
        return False


def _is_boundary_file_vectorized(boundary_filepath: str) -> bool:
    return str(os.path.splitext(boundary_filepath)).lower() in _VECTORIZED_FILENAMES


def _check_build_successful_and_built_data_config_sections_available(config: configs.Config) -> bool:
    filepath = _get_built_data_config_filepath(config)
    return os.path.exists(filepath)


def _check_built_data_files_exist(config: configs.Config) -> bool:
    filepaths = \
        _get_built_features_filepaths(config) + \
        _get_built_responses_filepaths(config) + \
        _get_built_weights_filepaths(config)
    missing_files = [filepath for filepath in filepaths if not os.path.exists(filepath)]
    if not missing_files:
        _logger.debug('Built data files found at paths: {}'.format(', '.join(filepaths)))
    else:
        _logger.warning('Built data files were not found at paths: {}'.format(', '.join(missing_files)))
    return not missing_files


################### File/Dataset Opening Functions ##############################


# TODO:  improve typing return
def _get_boundary_sets_from_boundary_files(config: configs.Config) -> List:
    if not config.raw_files.boundary_files:
        boundary_sets = [None] * len(config.raw_files.feature_files)
    else:
        boundary_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) if loc_file is not None else None
                         for loc_file in config.raw_files.boundary_files]
    return boundary_sets


def _create_munged_features_responses_data_files(config: configs.Config, num_features: int, num_responses: int) \
        -> Tuple[np.array, np.array]:
    basename = _get_memmap_basename(config)
    shape = [config.data_build.max_samples, config.data_build.window_radius * 2, config.data_build.window_radius * 2]
    shape_features = tuple(shape + [num_features])
    shape_responses = tuple(shape + [num_responses])

    features_filepath = _get_temporary_features_filepath(config)
    responses_filepath = _get_temporary_responses_filepath(config)

    _logger.debug('Create temporary munged features data file with shape {} at {}'.format(
        shape_features, features_filepath))
    features = np.memmap(features_filepath, dtype=np.float32, mode='w+', shape=shape_features)

    _logger.debug('Create temporary munged responses data file with shape {} at {}'.format(
        shape_responses, responses_filepath))
    responses = np.memmap(responses_filepath, dtype=np.float32, mode='w+', shape=shape_responses)
    return features, responses


def _create_temporary_weights_data_files(config: configs.Config, num_samples: int) -> np.array:
    weights_filepath = _get_temporary_weights_filepath(config)
    _logger.debug('Create temporary munged weights data file at {}'.format(weights_filepath))
    shape = tuple([num_samples, config.data_build.window_radius * 2, config.data_build.window_radius * 2, 1])
    return np.memmap(weights_filepath, dtype=np.float32, mode='w+', shape=shape)


def _load_built_data_files(config: configs.Config, writeable: bool = False) \
        -> Tuple[List[np.array], List[np.array], List[np.array]]:
    _logger.debug('Loading built data files with writeable == {}'.format(writeable))
    feature_files = _get_built_features_filepaths(config)
    response_files = _get_built_responses_filepaths(config)
    weight_files = _get_built_weights_filepaths(config)
    mode = 'r+' if writeable else 'r'

    features = [np.load(feature_file, mmap_mode=mode) for feature_file in feature_files]
    responses = [np.load(response_file, mmap_mode=mode) for response_file in response_files]
    weights = [np.load(weight_file, mmap_mode=mode) for weight_file in weight_files]
    _logger.debug('Built data files loaded')
    return features, responses, weights


################### Save/remove Functions ##############################


def _save_built_data_files(
        features_munged: np.array,
        responses_munged: np.array,
        weights_munged: np.array,
        config: configs.Config
) -> None:
    _logger.debug('Create fold assignments')
    fold_assignments = np.zeros(features_munged.shape[0]).astype(int)
    for f in range(0, config.data_build.number_folds):
        idx_start = int(round(f / config.data_build.number_folds * len(fold_assignments)))
        idx_finish = int(round((f + 1) / config.data_build.number_folds * len(fold_assignments)))
        fold_assignments[idx_start:idx_finish] = f

    _logger.debug('Save features to memmapped arrays separated by folds')
    features_filepaths = _get_built_features_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug('Save features fold {}'.format(idx_fold))
        np.save(features_filepaths[idx_fold], features_munged[fold_assignments == idx_fold, ...])

    _logger.debug('Save responses to memmapped arrays separated by folds')
    responses_filepaths = _get_built_responses_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug('Save responses fold {}'.format(idx_fold))
        np.save(responses_filepaths[idx_fold], responses_munged[fold_assignments == idx_fold, ...])

    _logger.debug('Save weights to memmapped arrays separated by folds')
    weights_filepaths = _get_built_weights_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug('Save weights fold {}'.format(idx_fold))
        np.save(weights_filepaths[idx_fold], weights_munged[fold_assignments == idx_fold, ...])


def _save_built_data_config_sections_to_verify_successful(config: configs.Config) -> None:
    filepath = _get_built_data_config_filepath(config)
    _logger.debug('Saving built data config sections to {}'.format(filepath))
    configs.save_config_to_file(config, filepath, include_sections=['raw_files', 'data_build'])


def _remove_temporary_data_files(config: configs.Config) -> None:
    _logger.debug('Remove temporary munge files')
    if os.path.exists(_get_temporary_features_filepath(config)):
        os.remove(_get_temporary_features_filepath(config))
    if os.path.exists(_get_temporary_responses_filepath(config)):
        os.remove(_get_temporary_responses_filepath(config))
    if os.path.exists(_get_temporary_weights_filepath(config)):
        os.remove(_get_temporary_weights_filepath(config))


################### Array Resizing Functions ##############################


def _resize_munged_features(features_munged: np.array, num_samples: int, config: configs.Config) -> np.array:
    _logger.debug('Resize memmapped features array with out-of-memory methods; original features shape {}'.format(
        features_munged.shape))
    new_features_shape = tuple([num_samples] + list(features_munged.shape[1:]))
    _logger.debug('Delete in-memory data to force data dump')
    del features_munged
    _logger.debug('Reload data from memmap files with modified sizes; new features shape {}'.format(
        new_features_shape))
    features_munged = np.memmap(
        _get_temporary_features_filepath(config), dtype=np.float32, mode='r+', shape=new_features_shape)
    return features_munged


def _resize_munged_responses(responses_munged: np.array, num_samples: int, config: configs.Config) -> np.array:
    _logger.debug('Resize memmapped responses array with out-of-memory methods; original responses shape {}'.format(
        responses_munged.shape))
    new_responses_shape = tuple([num_samples] + list(responses_munged.shape[1:]))
    _logger.debug('Delete in-memory data to force data dump')
    del responses_munged
    _logger.debug('Reload data from memmap files with modified sizes; new responses shape {}'.format(
        new_responses_shape))
    responses_munged = np.memmap(
        _get_temporary_responses_filepath(config), dtype=np.float32, mode='r+', shape=new_responses_shape)
    return responses_munged


################### Filepath Nomenclature Functions ##############################


def _create_built_data_output_directory(config: configs.Config) -> None:
    if not os.path.exists(config.data_build.dir_out):
        _logger.debug('Create built data output directory at {}'.format(config.data_build.dir_out))
        os.makedirs(config.data_build.dir_out)


def _get_temporary_features_filepath(config: configs.Config) -> str:
    return _get_temporary_data_filepaths(config, _FILENAME_FEATURES_TEMPORARY_SUFFIX)


def _get_temporary_responses_filepath(config: configs.Config) -> str:
    return _get_temporary_data_filepaths(config, _FILENAME_RESPONSES_TEMPORARY_SUFFIX)


def _get_temporary_weights_filepath(config: configs.Config) -> str:
    return _get_temporary_data_filepaths(config, _FILENAME_WEIGHTS_TEMPORARY_SUFFIX)


def _get_temporary_data_filepaths(config: configs.Config, filename_suffix: str) -> str:
    return _get_memmap_basename(config) + filename_suffix


def _get_built_features_filepaths(config: configs.Config) -> List[str]:
    return _get_built_data_filepaths(config, _FILENAME_FEATURES_SUFFIX)


def _get_built_responses_filepaths(config: configs.Config) -> List[str]:
    return _get_built_data_filepaths(config, _FILENAME_RESPONSES_SUFFIX)


def _get_built_weights_filepaths(config: configs.Config) -> List[str]:
    return _get_built_data_filepaths(config, _FILENAME_WEIGHTS_SUFFIX)


def _get_built_data_config_filepath(config: configs.Config) -> str:
    return _get_built_data_filepaths(config, _FILENAME_BUILT_DATA_CONFIG_SUFFIX)[0]


def _get_built_data_filepaths(config: configs.Config, filename_suffix: str) -> List[str]:
    basename = _get_memmap_basename(config)
    filepaths = [basename + filename_suffix.format(idx_fold) for idx_fold in range(config.data_build.number_folds)]
    return filepaths


def _get_memmap_basename(config: configs.Config) -> str:
    filepath_separator = config.data_build.filename_prefix_out + '_' if config.data_build.filename_prefix_out else ''
    return os.path.join(config.data_build.dir_out, filepath_separator)


################### Logging Functions ##############################


def _log_munged_data_information(
        features_munged: np.array = None,
        responses_munged: np.array = None,
        weights_munged: np.array = None
) -> None:
    if features_munged is not None:
        _logger.info('Munged features shape: {}'.format(features_munged.shape))
    if responses_munged is not None:
        _logger.info('Munged responses shape: {}'.format(responses_munged.shape))
    if weights_munged is not None:
        _logger.info('Munged weights shape: {}'.format(weights_munged.shape))
