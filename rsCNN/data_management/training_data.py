import gdal
import os
from pathlib import Path
import re
from tqdm import tqdm
from typing import List

import fiona
import numpy as np
import numpy.matlib
import ogr
import rasterio.features
from rsCNN.utils import logging
# TODO:  remove * imports
from rsCNN.utils.general import *
from rsCNN.data_management import scalers, DataConfig


_logger = logging.get_child_logger(__name__)

MAX_UNIQUE_RESPONSES = 100


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


def build_or_load_rawfile_data(config, rebuild=False):

    data_container = Dataset(config)
    data_container.check_input_files(config.raw_feature_file_list,
                                     config.raw_response_file_list, config.boundary_file_list)

    data_container.feature_raw_band_types = data_container.get_band_types(
        config.raw_feature_file_list, config.feature_raw_band_type_input)
    data_container.response_raw_band_types = data_container.get_band_types(
        config.raw_response_file_list, config.response_raw_band_type_input)

    if (rebuild is False):
        features, responses, weights, read_success = open_memmap_files(config)

    if (read_success is False or rebuild is True):
        assert config.raw_feature_file_list is not [], 'feature files to pull data from are required'
        assert config.raw_response_file_list is not [], 'response files to pull data from are required'

        if (config.ignore_projections is False):
            check_projections(config.raw_feature_file_list,
                              config.raw_response_file_list,
                              config.boundary_file_list)

        # TODO: deal with boundary file list here as well if it exists
        check_resolutions(config.raw_feature_file_list,
                          config.raw_response_file_list)

        if (config.response_data_format == 'FCN'):
            features, responses, weights, response_band_types = build_training_data_ordered(config,
                                                                                            data_container.feature_raw_band_types,
                                                                                            data_container.response_raw_band_types)
        elif (config.response_data_format == 'CNN'):
            features, responses, weights, response_band_types = build_training_data_from_response_points(config,
                                                                                                         data_container.feature_raw_band_types,
                                                                                                         data_container.response_raw_band_types)
        else:
            raise NotImplementedError('Unknown response data format')

    data_container.features = features
    data_container.responses = responses
    data_container.weights = weights
    data_container.response_band_types = weights

    return data_container


def open_memmap_files(config, writeable=False, override_success_file=False):

    success = True
    if (os.path.isfile(config.successful_data_save_file) is not True and override_success_file is not True):
        _logger.debug('no saved data')
        success = False

    if (writeable is True):
        mode = 'r+'
    else:
        mode = 'r'

    features = []
    responses = []
    weights = []
    # TODO:  Phil:  I get a failed read on the first file when running this and building... but I think everything works
    #  after that? I don't know if "failed read" is the right message for something that succeeds?
    # FABINA - if you get a failed read message, than you should also only be returning None...is that not the case?
    for fold in range(config.n_folds):
        if (os.path.isfile(config.feature_files[fold])):
            features.append(np.load(config.feature_files[fold], mmap_mode=mode))
        else:
            success = False
            _logger.debug('failed read at {}'.format(config.feature_files[fold]))
            break
        if (os.path.isfile(config.response_files[fold])):
            responses.append(np.load(config.response_files[fold], mmap_mode=mode))
        else:
            _logger.debug('failed read at {}'.format(config.response_files[fold]))
            success = False
            break
        if (os.path.isfile(config.weight_files[fold])):
            weights.append(np.load(config.weight_files[fold], mmap_mode=mode))
        else:
            _logger.debug('failed read at {}'.format(config.weight_files[fold]))
            success = False
            break

    if (success):
        return features, responses, weights, True
    else:
        return None, None, None, False


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


def check_projections(a_files, b_files, c_files=[]):

    loc_a_files = [item for sublist in a_files for item in sublist]
    loc_b_files = [item for sublist in b_files for item in sublist]
    if (len(c_files) > 0):
        loc_c_files = [item for sublist in c_files for item in sublist]
    else:
        loc_c_files = []

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
            assert (a_proj[_p] == b_proj[_p] and a_proj == c_proj[_p]), 'Projection_mismatch between\n{}: {},\n{}: {},\n{}: {}'.\
                format(a_proj[_p], loc_a_files[_p], b_proj[_p], loc_b_files[_p], c_proj[_p], loc_c_files[_p])
        else:
            assert (a_proj[_p] == b_proj[_p]), 'Projection_mismatch between\n{}: {}\n{}: {}'.\
                format(a_proj[_p], loc_a_files[_p], b_proj[_p], loc_b_files[_p])


def check_resolutions(a_files, b_files, c_files=[]):

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
            assert (np.all(a_res[_p] == b_res[_p]) and np.all(a_res == c_res[_p])), 'Resolution mismatch between\n{}: {},\n{}: {},\n{}: {}'.\
                format(a_res[_p], loc_a_files[_p], b_res[_p], loc_b_files[_p], c_res[_p], loc_c_files[_p])
        else:
            assert (np.all(a_res[_p] == b_res[_p])), 'Resolution mimatch between\n{}: {}\n{}: {}'.\
                format(a_res[_p], loc_a_files[_p], b_res[_p], loc_b_files[_p])


# deprecated, keeping for potential future use
def check_data_extents_and_projections(set_a, set_b, set_b_is_vector=False, set_c=[], set_c_is_vector=[], ignore_projections=False, ignore_extents=False):
    """ Check to see if two different gdal datasets have the same projection, geotransform, and extent.
    Arguments:
    set_a - list
      First list of gdal datasets to check.
    set_b - list
      Second list of gdal datasets (or vectors) to check.

    Keyword Arguments:
    set_b_is_vector - boolean
      Flag to indicate if set_b is a vector, as opposed to a gdal_dataset.
    set_c - list
      A third (optional) list of gdal datasets to check.
    set_c_is_vector - list
      List of flags to indicate if set_c is a vector, as opposed to a gdal_dataset.
    ignore_projections - boolean
      A flag to ignore projection differences between feature and response sets - use only if you 
      are sure the projections are really the same.


    Return: 
    None, simply throw error if the check fails
    """
    # TODO:  this should confirm that the input is actually a list
    if (len(set_a) != len(set_b)):
        raise Exception('different number of training features and responses')
    if (len(set_c) > 0):
        if (len(set_a) != len(set_c)):
            raise Exception('different number of training features and boundary files - give None for blank boundary')

    for n in range(0, len(set_a)):
        a_proj = get_proj(set_a[n], False)
        b_proj = get_proj(set_b[n], set_b_is_vector)

        if (a_proj != b_proj and ignore_projections is False):
            raise Exception(('projection mismatch between', set_a[n], 'and', set_b[n]))

        if (len(set_c) > 0):
            if (set_c[n] is not None):
                c_proj = get_proj(set_c[n], set_c_is_vector)
            else:
                c_proj = b_proj

            if (a_proj != c_proj and ignore_projections is False):
                raise Exception(('projection mismatch between', set_a[n], 'and', set_c[n]))

        if (set_b_is_vector == False):
            dataset_a = gdal.Open(set_a[n], gdal.GA_ReadOnly)
            dataset_b = gdal.Open(set_b[n], gdal.GA_ReadOnly)
            a_trans = dataset_a.GetGeoTransform()
            b_trans = dataset_b.GetGeoTransform()

            if (dataset_a.GetProjection() != dataset_b.GetProjection() and ignore_projections is False):
                raise Exception(('projection mismatch between', set_a[n], 'and', set_b[n]))

            if (a_trans[1] != b_trans[1] or a_trans[5] != b_trans[5]):
                raise Exception(('resolution mismatch between', set_a[n], 'and', set_b[n]))

            if (ignore_extents is False):
                if (a_trans[0] != b_trans[0] or a_trans[3] != b_trans[3]):
                    raise Exception(('upper left mismatch between', set_a[n], 'and', set_b[n]))

                if (dataset_a.RasterXSize != dataset_b.RasterXSize or dataset_a.RasterYSize != dataset_b.RasterYSize):
                    raise Exception(('extent mismatch between', set_a[n], 'and', set_b[n]))

        if (len(set_c) > 0):
            if (set_c[n] is not None and set_c_is_vector[n] is False):
                dataset_a = gdal.Open(set_a[n], gdal.GA_ReadOnly)
                dataset_c = gdal.Open(set_c[n], gdal.GA_ReadOnly)
                a_trans = dataset_a.GetGeoTransform()
                c_trans = dataset_c.GetGeoTransform()

                if (dataset_a.GetProjection() != dataset_c.GetProjection() and ignore_projections == False):
                    raise Exception(('projection mismatch between', set_a[n], 'and', set_c[n]))

                if (a_trans[1] != c_trans[1] or a_trans[5] != c_trans[5]):
                    raise Exception(('resolution mismatch between', set_a[n], 'and', set_c[n]))

                if (ignore_extents is False):
                    if (a_trans[0] != c_trans[0] or a_trans[3] != c_trans[3]):
                        raise Exception(('upper left mismatch between', set_a[n], 'and', set_c[n]))

                    if (dataset_a.RasterXSize != dataset_c.RasterXSize or dataset_a.RasterYSize != dataset_c.RasterYSize):
                        raise Exception(('extent mismatch between', set_a[n], 'and', set_c[n]))

# Calculates categorical weights for a single response


def calculate_categorical_weights(responses, weights, config, batch_size=100):

    # find upper and lower boud
    lb = config.window_radius - config.internal_window_radius
    ub = -lb

    # get response/total counts (batch-wise)
    response_counts = np.zeros(responses[0].shape[-1])
    total_valid_count = 0
    for _array in range(len(responses)):
        if (_array is not config.validation_fold and _array is not config.test_fold):
            for ind in range(0, responses[_array].shape[0], batch_size):
                if (lb == 0):
                    lr = (responses[_array])[ind:ind+batch_size, ...]
                else:
                    lr = (responses[_array])[ind:ind+batch_size, lb:ub, lb:ub, :]
                lr[lr == config.response_nodata_value] = np.nan
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


def read_mask_chunk(boundary_vector_file: str, boundary_subset_geotransform: tuple, b_set: tuple, boundary_upper_left: List, window_diameter: int, boundary_bad_value: float):
    # Start by checking if we're inside boundary, if there is one
    mask = None
    if (boundary_vector_file is not None):
        mask = rasterize_vector(boundary_vector_file, boundary_subset_geotransform, (window_diameter, window_diameter))
    if (b_set is not None):
        mask = b_set.ReadAsArray(boundary_upper_left[0], boundary_upper_left[1], window_diameter, window_diameter)

    if mask is None:
        mask = np.zeros((window_diameter, window_diameter)).astype(bool)
    else:
        mask = mask == boundary_bad_value

    return mask


def read_map_subset(datasets: List, upper_lefts: List[List[int]], window_diameter: int, mask, nodata_value):
    # Next check to see if we have a response, if so read all
    local_array = np.zeros((window_diameter, window_diameter, np.sum([lset.RasterCount for lset in datasets])))
    idx = 0
    for _file in range(len(datasets)):
        file_set = datasets[_file]
        file_upper_left = upper_lefts[_file]
        file_array = np.zeros((window_diameter, window_diameter, file_set.RasterCount))
        for _b in range(file_set.RasterCount):
            file_array[:, :, _b] = file_set.GetRasterBand(
                _b+1).ReadAsArray(file_upper_left[0], file_upper_left[1], window_diameter, window_diameter)

        file_array[file_array == nodata_value] = np.nan
        file_array[np.isfinite(file_array) is False] = np.nan
        file_array[mask, :] = np.nan

        mask[np.any(np.isnan(file_array), axis=-1)] = True
        if np.all(mask):
            return None, None
        local_array[..., idx:idx+file_array.shape[-1]] = file_array
        idx += file_array.shape[-1]

    return local_array, mask


def read_labeling_chunk(f_sets: List[tuple],
                        feature_upper_lefts: List[List[int]],
                        config: DataConfig,
                        boundary_vector_file: str = None,
                        boundary_subset_geotransform: tuple = None,
                        b_set=None,
                        boundary_upper_left: List[int] = None):

    for _f in range(len(feature_upper_lefts)):
        if (np.any(feature_upper_lefts[_f] < config.window_radius)):
            _logger.trace('Feature read OOB')
            return None
        if (feature_upper_lefts[_f][0] > f_sets[_f].RasterXSize - config.window_radius):
            _logger.trace('Feature read OOB')
            return None
        if (feature_upper_lefts[_f][1] > f_sets[_f].RasterYSize - config.window_radius):
            _logger.trace('Feature read OOB')
            return None

    window_diameter = config.window_radius * 2

    mask = read_mask_chunk(boundary_vector_file,
                           boundary_subset_geotransform,
                           b_set,
                           boundary_upper_left,
                           window_diameter,
                           config.boundary_bad_value)

    if not _check_mask_data_sufficient(mask, config.nodata_maximum_fraction):
        _logger.trace('Insufficient mask data')
        return None

    local_feature, mask = read_map_subset(f_sets, feature_upper_lefts,
                                          window_diameter, mask, config.feature_nodata_value)

    if not _check_mask_data_sufficient(mask, config.nodata_maximum_fraction):
        _logger.trace('Insufficient feature data')
        return None

    # Final check (propogate mask forward), and return
    local_feature[mask, :] = np.nan

    return local_feature


def read_segmentation_chunk(f_sets: List[tuple],
                            r_sets: List[tuple],
                            feature_upper_lefts: List[List[int]],
                            response_upper_lefts: List[List[int]],
                            config: DataConfig,
                            boundary_vector_file: str = None,
                            boundary_subset_geotransform: tuple = None,
                            b_set=None,
                            boundary_upper_left: List[int] = None):
    window_diameter = config.window_radius * 2

    mask = read_mask_chunk(boundary_vector_file,
                           boundary_subset_geotransform,
                           b_set,
                           boundary_upper_left,
                           window_diameter,
                           config.boundary_bad_value)
    mv = [np.sum(mask)]

    if not _check_mask_data_sufficient(mask, config.nodata_maximum_fraction):
        return None, None

    local_response, mask = read_map_subset(r_sets, response_upper_lefts,
                                           window_diameter, mask, config.response_nodata_value)
    if not _check_mask_data_sufficient(mask, config.nodata_maximum_fraction):
        return None, None
    mv.append(np.sum(mask))

    if (config.response_min_value is not None):
        local_response[local_response < config.response_min_value] = np.nan
    if (config.response_max_value is not None):
        local_response[local_response > config.response_max_value] = np.nan
    mask[np.any(np.isnan(local_response), axis=-1)] = True
    mv.append(np.sum(mask))

    if (mask is None):
        return None, None
    if not _check_mask_data_sufficient(mask, config.nodata_maximum_fraction):
        return None, None

    local_feature, mask = read_map_subset(f_sets, feature_upper_lefts,
                                          window_diameter, mask, config.feature_nodata_value)
    mv.append(np.sum(mask))

    if not _check_mask_data_sufficient(mask, config.nodata_maximum_fraction):
        return None, None

    # Final check (propogate mask forward), and return
    local_feature[mask, :] = np.nan
    local_response[mask, :] = np.nan

    return local_feature, local_response


class Dataset:

    """ A container class that holds all sorts of data objects
    """

    def __init__(self, config: DataConfig):
        self.features = []
        self.responses = []
        self.weights = []

        self.feature_band_types = None
        self.response_band_types = None
        self.feature_raw_band_types = None
        self.response_raw_band_types = None

        self.feature_scalers = []
        self.response_scalers = []

        self.train_folds = None

        self.config = config

    def build_or_load_scalers(self, rebuild=False):

        data_config = self.config

        feat_scaler_atr = {'savename_base': data_config.data_save_name + '_feature_scaler'}
        feature_scaler = scalers.get_scaler(data_config.feature_scaler_name, feat_scaler_atr)
        resp_scaler_atr = {'savename_base': data_config.data_save_name + '_response_scaler'}
        response_scaler = scalers.get_scaler(data_config.response_scaler_name, resp_scaler_atr)
        feature_scaler.load()
        response_scaler.load()

        self.train_folds = [x for x in range(data_config.n_folds)
                            if x is not data_config.validation_fold and x is not data_config.test_fold]

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
        # file list b is a list for each site, with one boundary file expected to be the interior boundary for all bands.

        # Check that feature and response files are lists
        assert type(f_file_list) is list, 'Feature files must be a list of lists'
        assert type(r_file_list) is list, 'Response files must be a list of lists'

        # Checks on the matching numbers of sites
        assert len(f_file_list) == len(r_file_list), 'Feature and response site lists must be the same length'
        assert len(f_file_list) > 0, 'At least one feature and response site is required'
        if (len(b_file_list) > 0):
            assert len(b_file_list) == len(f_file_list), 'Boundary and feature site lists must be the same length'

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
            assert len(
                f_file_list[_site]) == num_f_files_per_site, 'Inconsistent number of feature files at site {}'.format(_site)
            assert len(
                r_file_list[_site]) == num_r_files_per_site, 'Inconsistent number of response files at site {}'.format(_site)

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
        # 2) band_types is a list of strings within valid_band_types - assume each band from the associated file is the specified type,
        #    requires len(band_types) == len(file_list[0])
        # 3) band_types is list of lists (of strings, contained in valid_band_types), with the outer list referring to
        #    files and the inner list referring to bands

        num_bands_per_file = [gdal.Open(x, gdal.GA_ReadOnly).RasterCount for x in file_list[0]]

        # Nonetype, option 1 from above, auto-generate
        if (band_types is None):
            for _file in range(len(file_list[0])):
                output_raw_band_types = []
                output_raw_band_types.append(['R' for _band in range(num_bands_per_file[_file])])

        else:
            assert type(band_types) is list, 'band_types must be None or a list'

            # List of lists, option 3 from above - just check components
            if (type(band_types[0]) is list):
                for _file in range(len(band_types)):
                    assert type(
                        band_types[_file]) is list, 'If one element of band_types is a list, all elements must be lists'
                    assert len(
                        band_types[_file]) == num_bands_per_file[_file], 'File {} has wrong number of band types'.format(_file)
                    for _band in range(len(band_types[_file])):
                        assert band_types[_file][_band] in valid_band_types, 'Invalid band types at file {}, band {}'.format(
                            _file, _band)

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


def upper_left_pixel(trans, interior_x, interior_y):
    x_ul = max((trans[0] - interior_x)/trans[1], 0)
    y_ul = max((interior_y - trans[3])/trans[5], 0)
    return x_ul, y_ul


def get_interior_rectangle(dataset_list_of_lists: List[List[tuple]]):

    # Convert list of lists or list for interior convenience
    dataset_list = [item for sublist in dataset_list_of_lists for item in sublist]

    # Get list of all gdal geotransforms
    trans_list = []
    for _d in range(len(dataset_list)):
        trans_list.append(dataset_list[_d].GetGeoTransform())

    # Find the interior (UL) x,y coordinates in map-space
    interior_x = np.nanmax([x[0] for x in trans_list])
    interior_y = np.nanmin([x[3] for x in trans_list])

    # calculate the UL coordinates in pixel-space
    ul_list = []
    for _d in range(len(dataset_list)):
        ul_list.append(list(upper_left_pixel(trans_list[_d], interior_x, interior_y)))

    # calculate the size of the matched interior extent
    x_len = int(np.floor(np.min([dataset_list[_d].RasterXSize - ul_list[_d][0] for _d in range(len(dataset_list))])))
    y_len = int(np.floor(np.min([dataset_list[_d].RasterYSize - ul_list[_d][1] for _d in range(len(dataset_list))])))

    # separate out into list of lists for return
    return_ul_list = []
    idx = 0
    for _l in range(len(dataset_list_of_lists)):
        local_list = []
        for _d in range(len(dataset_list_of_lists[_l])):
            local_list.append(ul_list[idx])
            idx += 1
        local_list = np.array(local_list)
        return_ul_list.append(local_list)

    return return_ul_list, x_len, y_len


# def get_interior_rectangle(feature_set, response_set, boundary_set):
#    f_trans = feature_set.GetGeoTransform()
#    r_trans = response_set.GetGeoTransform()
#    b_trans = None
#    if (boundary_set is not None):
#        b_trans = boundary_set.GetGeoTransform()
#
#    # Calculate the interior space location and extent
#
#    # Find the interior (UL) x,y coordinates in map-space
#    interior_x = max(r_trans[0], f_trans[0])
#    interior_y = min(r_trans[3], f_trans[3])
#    if (b_trans is not None):
#        interior_x = max(interior_x, b_trans[0])
#        interior_y = max(interior_y, b_trans[3])
#
#    # calculate the feature and response UL coordinates in pixel-space
#    f_x_ul, f_y_ul = upper_left_pixel(f_trans, interior_x, interior_y)
#    r_x_ul, r_y_ul = upper_left_pixel(r_trans, interior_x, interior_y)
#
#    # calculate the size of the matched interior extent
#    x_len = min(feature_set.RasterXSize - f_x_ul, response_set.RasterXSize - r_x_ul)
#    y_len = min(feature_set.RasterYSize - f_y_ul, response_set.RasterYSize - r_y_ul)
#
#    # update the UL location, and the interior extent, if there is a boundary
#    if (b_trans is not None):
#        b_x_ul, b_y_ul = upper_left_pixel(b_trans, interior_x, interior_y)
#        x_len = min(x_len, boundary_set.RasterXSize - b_x_ul)
#        y_len = min(y_len, boundary_set.RasterYSize - b_y_ul)
#
#    # convert these UL coordinates to an array for easy addition later
#    f_ul = np.array([f_x_ul, f_y_ul])
#    r_ul = np.array([r_x_ul, r_y_ul])
#    if (b_trans is not None):
#        b_ul = np.array([b_x_ul, b_y_ul])
#    else:
#        b_ul = None
#
#    return f_ul, r_ul, b_ul, x_len, y_len


def one_hot_encode_array(raw_band_types, array, memmap_file):

    cat_band_locations = [idx for idx, val in enumerate(raw_band_types) if val == 'C']
    band_types = raw_band_types.copy()
    for _c in reversed(range(len(cat_band_locations))):

        un_array = array[..., cat_band_locations[_c]]
        un_array = np.unique(un_array[np.isfinite(un_array)])
        assert len(un_array) < MAX_UNIQUE_RESPONSES,\
            'Too many ({}) unique responses found, suspected incorrect categorical specification'.format(len(un_array))
        _logger.info('Found {} categorical responses'.format(len(un_array)))
        _logger.debug('Cat response: {}'.format(un_array))

        array_shape = list(array.shape)
        array_shape[-1] = len(un_array) + array.shape[-1] - 1

        cat_memmap_file = os.path.join(os.path.dirname(
            memmap_file), os.path.basename(memmap_file).split('.')[0] + '_cat.npy')
        cat_array = np.memmap(cat_memmap_file,
                              dtype=np.float32,
                              mode='w+',
                              shape=tuple(array_shape))

        # One hot-encode
        for _r in range(array_shape[-1]):
            if (_r >= cat_band_locations[_c] and _r < len(un_array)):
                cat_array[..., _r] = np.squeeze(array[..., cat_band_locations[_c]] ==
                                                un_array[_r - cat_band_locations[_c]])
            else:
                if (_r < cat_band_locations[_c]):
                    cat_array[..., _r] = array[..., _r]
                else:
                    cat_array[..., _r] = array[..., _r - len(un_array) + 1]

        # Force file dump, and then reload the encoded responses as the primary response
        del array, cat_array
        if (os.path.isfile(memmap_file)):
            os.remove(memmap_file)
        memmap_file = cat_memmap_file
        array = np.memmap(memmap_file, dtype=np.float32, mode='r+', shape=tuple(array_shape))

        band_types.pop(cat_band_locations[_c])
        for _r in range(len(un_array)):
            band_types.insert(cat_band_locations[_c], 'B' + str(int(_c)))
    return array, band_types


def build_training_data_ordered(config: DataConfig, feature_raw_band_types: List[List[str]], response_raw_band_types: List[List[str]]):

    if (config.random_seed is not None):
        np.random.seed(config.random_seed)

    if (isinstance(config.max_samples, list)):
        if (len(config.max_samples) != len(config.raw_feature_file_list)):
            raise Exception('max_samples must equal feature_file_list length, or be an integer.')

    n_features = np.sum([len(feat_type) for feat_type in feature_raw_band_types])
    n_responses = np.sum([len(resp_type) for resp_type in response_raw_band_types])

    feature_memmap_file = config.data_save_name + '_feature_munge_memmap.npy'
    response_memmap_file = config.data_save_name + '_response_munge_memmap.npy'
    weight_memmap_file = config.data_save_name + '_weight_munge_memmap.npy'

    # TODO: fix max size issue, but force for now to prevent overly sized sets
    assert config.max_samples * (config.window_radius*2)**2 * n_features / 1024.**3 < 10, 'max_samples too large'
    features = np.memmap(feature_memmap_file,
                         dtype=np.float32,
                         mode='w+',
                         shape=(config.max_samples, config.window_radius*2, config.window_radius*2, n_features))

    responses = np.memmap(response_memmap_file,
                          dtype=np.float32,
                          mode='w+',
                          shape=(config.max_samples, config.window_radius*2, config.window_radius*2, n_responses))

    sample_index = 0
    boundary_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly)
                     if loc_file is not None else None for loc_file in config.boundary_file_list]
    if (len(boundary_sets) == 0):
        boundary_sets = [None for i in range(len(config.raw_feature_file_list))]
    for _site in range(0, len(config.raw_feature_file_list)):

        # open requisite datasets
        feature_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_feature_file_list[_site]]
        response_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_response_file_list[_site]]

        # Calculate the interior space location and extent
        [f_ul, r_ul, b_ul], x_len, y_len = get_interior_rectangle(
            [feature_sets, response_sets, [bs for bs in boundary_sets if bs is not None]])

        # Use interior space calculations to calculate pixel-based interior space offsets for data aquisition
        collist = [x for x in range(0,
                                    int(x_len - 2*config.window_radius),
                                    int(config.internal_window_radius*2))]
        rowlist = [y for y in range(0,
                                    int(y_len - 2*config.window_radius),
                                    int(config.internal_window_radius*2))]

        colrow = np.zeros((len(collist)*len(rowlist), 2)).astype(int)
        colrow[:, 0] = np.matlib.repmat(np.array(collist).reshape((-1, 1)), 1, len(rowlist)).flatten()
        colrow[:, 1] = np.matlib.repmat(np.array(rowlist).reshape((1, -1)), len(collist), 1).flatten()
        del collist, rowlist

        colrow = colrow[np.random.permutation(colrow.shape[0]), :]

        ref_trans = feature_sets[0].GetGeoTransform()
        subset_geotransform = None
        if (len(config.boundary_file_list) > 0):
            if (config.boundary_file_list[_site] is not None and config.boundary_as_vectors[_site]):
                subset_geotransform = [ref_trans[0], ref_trans[1], 0, ref_trans[3], 0, ref_trans[5]]

        for _cr in tqdm(range(len(colrow)), ncols=80):

            # Determine local information about boundary file
            local_boundary_vector_file = None
            local_boundary_upper_left = None
            if (boundary_sets[_site] is not None):
                local_boundary_upper_left = b_ul + colrow[_cr, :]
            if (subset_geotransform is not None):
                subset_geotransform[0] = ref_trans[0] + (f_ul[0][0] + colrow[_cr, 0]) * ref_trans[1]
                subset_geotransform[3] = ref_trans[3] + (f_ul[0][1] + colrow[_cr, 1]) * ref_trans[5]
                local_boundary_vector_file = config.boundary_file_list[_site]

            local_feature, local_response = read_segmentation_chunk(feature_sets,
                                                                    response_sets,
                                                                    f_ul + colrow[_cr, :],
                                                                    r_ul + colrow[_cr, :],
                                                                    config,
                                                                    boundary_vector_file=local_boundary_vector_file,
                                                                    boundary_upper_left=local_boundary_upper_left,
                                                                    b_set=boundary_sets[_site],
                                                                    boundary_subset_geotransform=subset_geotransform)

            if (local_feature is not None):
                features[sample_index, ...] = local_feature.copy()
                responses[sample_index, ...] = local_response.copy()
                sample_index += 1
                if (sample_index >= config.max_samples):
                    break

    # Get the feature/response shapes for re-reading (modified ooc resize)
    feat_shape = list(features.shape)
    resp_shape = list(responses.shape)
    feat_shape[0] = sample_index
    resp_shape[0] = sample_index

    # Delete and reload feauters/responses, as a hard and fast way to force data dump to disc and reload
    # with a modified size....IE, an ooc resize
    del features, responses
    features = np.memmap(feature_memmap_file, dtype=np.float32, mode='r+', shape=(tuple(feat_shape)))
    responses = np.memmap(response_memmap_file, dtype=np.float32, mode='r+', shape=tuple(resp_shape))

    # Shuffle the data one last time (in case the fold-assignment would otherwise be biased beacuase of
    # the feature/response file order
    perm = np.random.permutation(features.shape[0])
    features = features[perm, :]
    responses = responses[perm, :]
    del perm

    fold_assignments = np.zeros(responses.shape[0]).astype(int)
    for f in range(0, config.n_folds):
        idx_start = int(round(f / config.n_folds * len(fold_assignments)))
        idx_finish = int(round((f + 1) / config.n_folds * len(fold_assignments)))
        fold_assignments[idx_start:idx_finish] = f

    # Set up initial weights....will add in class-balancing if appropriate later
    weights = np.memmap(weight_memmap_file,
                        dtype=np.float32,
                        mode='w+',
                        shape=(features.shape[0], features.shape[1], features.shape[2], 1))
    weights[:, :, :, :] = 1
    weights[np.isnan(responses[..., 0])] = 0

    if (config.internal_window_radius != config.window_radius):
        buf = config.window_radius - config.internal_window_radius
        weights[:, :buf, :, -1] = 0
        weights[:, -buf:, :, -1] = 0
        weights[:, :, :buf, -1] = 0
        weights[:, :, -buf:, -1] = 0

    _logger.info('Feature shape: {}'.format(features.shape))
    _logger.info('Response shape: {}'.format(responses.shape))
    _logger.info('Weight shape: {}'.format(weights.shape))

    # one hot encode
    responses, response_band_types = one_hot_encode_array(response_raw_band_types, responses, response_memmap_file)

    for fold in range(config.n_folds):
        np.save(config.feature_files[fold], features[fold_assignments == fold, ...])
        np.save(config.response_files[fold], responses[fold_assignments == fold, ...])
        np.save(config.weight_files[fold], weights[fold_assignments == fold, ...])

    del features, responses, weights
    if ('C' in response_raw_band_types):
        if (np.sum(np.array(response_raw_band_types) == 'C') > 1):
            _logger.warning('Currently weighting is only enabled for one categorical response variable')
        features, responses, weights, success = open_memmap_files(config, writeable=True, override_success_file=True)
        weights = calculate_categorical_weights(responses, weights, config)
        del features, responses, weights

    # clean up munge files
    if (os.path.isfile(feature_memmap_file)):
        os.remove(feature_memmap_file)
    if (os.path.isfile(weight_memmap_file)):
        os.remove(weight_memmap_file)

    Path(config.successful_data_save_file).touch()
    features, responses, weights, success = open_memmap_files(config, writeable=False)
    return features, responses, weights, response_band_types


def build_training_data_from_response_points(config: DataConfig, feature_raw_band_types: List[List[str]], response_raw_band_types: List[List[str]]):

    if (config.random_seed is not None):
        np.random.seed(config.random_seed)

    colrow_per_site = []
    responses_per_site = []
    boundary_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly)
                     if loc_file is not None else None for loc_file in config.boundary_file_list]
    if (len(boundary_sets) == 0):
        boundary_sets = [None for i in range(len(config.raw_feature_file_list))]
    for _site in range(0, len(config.raw_feature_file_list)):
        # open requisite datasets
        feature_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_feature_file_list[_site]]
        response_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_response_file_list[_site]]

        # Calculate the interior space location and extent
        [f_ul, r_ul, b_ul], x_len, y_len = get_interior_rectangle(
            [feature_sets, response_sets, [bs for bs in boundary_sets if bs is not None]])

        collist = []
        rowlist = []
        responses = []
        # Run through first response
        for _line in range(y_len):
            line_dat = np.squeeze(response_sets[_site].ReadAsArray(r_ul[0][0], r_ul[0][1] + _line, x_len, 1))
            if (len(line_dat.shape) == 1):
                line_dat = line_dat.reshape(-1, 1)

            if (config.response_background_value is not None):
                good_data = np.all(line_dat != config.response_background_value, axis=1)
            else:
                good_data = np.ones(line_dat.shape[0]).astype(bool)

            if (config.response_nodata_value is not None):
                good_data[np.any(line_dat == config.response_nodata_value, axis=1)] = False

            if (np.sum(good_data) > 0):
                line_x = np.arange(x_len)
                line_y = line_x.copy()
                line_y[:] = _line

                collist.extend(line_x[good_data].tolist())
                rowlist.extend(line_y[good_data].tolist())
                responses.append(line_dat[good_data, :])

        colrow = np.vstack([np.array(collist), np.array(rowlist)]).T
        responses = np.vstack(responses).astype(np.float32)
        responses_per_file = [responses.copy()]

        for _file in range(1, len(response_sets)):
            responses = np.zeros(responses_per_file[0].shape[0])
            for _point in range(len(colrow)):
                responses[_point] = response_sets[_file].ReadAsArray(
                    r_ul[_file][0], r_ul[_file][1], 1, 1).astype(np.float32)
            responses_per_file.append(responses.copy())

        responses_per_file = np.hstack(responses_per_file)

        good_dat = np.all(responses_per_file != config.response_background_value, axis=1)
        responses_per_file = responses_per_file[good_dat, :]
        colrow = colrow[good_dat, :]

        colrow_per_site.append(colrow)
        responses_per_site.append(np.vstack(responses))

    total_samples = 0
    for _site in range(0, len(responses_per_site)):
        total_samples += responses_per_site[_site].shape[0]

    assert config.max_samples > 0, 'need more than 1 valid sample...'

    _logger.debug('total samples: {}'.format(total_samples))
    if (total_samples > config.max_samples):
        for _site in range(0, len(responses_per_site)):
            perm = np.random.permutation(len(responses_per_site[_site]))[:int(
                config.max_samples*len(responses_per_site[_site])/float(total_samples))]
            responses_per_site[_site] = responses_per_site[_site][perm, :]
            colrow_per_site[_site] = colrow_per_site[_site][perm, :]
            _logger.debug('perm len: {}'.format(len(perm)))

    total_samples = 0
    for _site in range(0, len(responses_per_site)):
        total_samples += responses_per_site[_site].shape[0]
    _logger.debug('total samples after trim: {}'.format(total_samples))

    n_features = np.sum([len(feat_type) for feat_type in feature_raw_band_types])

    # TODO: fix max size issue, but force for now to prevent overly sized sets
    feature_memmap_file = config.data_save_name + '_feature_munge_memmap.npy'
    response_memmap_file = config.data_save_name + '_response_munge_memmap.npy'
    assert total_samples * (config.window_radius*2)**2 * n_features / 1024.**3 < 10, 'max_samples too large'
    features = np.memmap(feature_memmap_file,
                         dtype=np.float32,
                         mode='w+',
                         shape=(total_samples, config.window_radius*2, config.window_radius*2, n_features))

    sample_index = 0
    boundary_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly)
                     if loc_file is not None else None for loc_file in config.boundary_file_list]
    if (len(boundary_sets) == 0):
        boundary_sets = [None for i in range(len(config.raw_feature_file_list))]
    for _site in range(0, len(config.raw_feature_file_list)):

        # open requisite datasets
        feature_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_feature_file_list[_site]]
        response_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_response_file_list[_site]]

        # Calculate the interior space location and extent
        [f_ul, r_ul, b_ul], x_len, y_len = get_interior_rectangle(
            [feature_sets, response_sets, [bs for bs in boundary_sets if bs is not None]])

        # colrow is current the response cetners, but we need to use the pixel ULs.  So subtract
        # out the corresponding feature radius
        colrow = colrow_per_site[_site] - config.window_radius

        ref_trans = feature_sets[0].GetGeoTransform()
        subset_geotransform = None
        if (len(config.boundary_file_list) > 0):
            if (config.boundary_file_list[_site] is not None and config.boundary_as_vectors[_site]):
                subset_geotransform = [ref_trans[0], ref_trans[1], 0, ref_trans[3], 0, ref_trans[5]]

        good_response_data = np.zeros(responses_per_site[_site].shape[0]).astype(bool)
        # Now read in features
        for _cr in tqdm(range(len(colrow)), ncols=80):

            # Determine local information about boundary file
            local_boundary_vector_file = None
            local_boundary_upper_left = None
            if (boundary_sets[_site] is not None):
                local_boundary_upper_left = b_ul + colrow[_cr, :]
            if (subset_geotransform is not None):
                subset_geotransform[0] = ref_trans[0] + (f_ul[0][0] + colrow[_cr, 0]) * ref_trans[1]
                subset_geotransform[3] = ref_trans[3] + (f_ul[0][1] + colrow[_cr, 1]) * ref_trans[5]
                local_boundary_vector_file = config.boundary_file_list[_site]

            local_feature = read_labeling_chunk(feature_sets,
                                                f_ul + colrow[_cr, :],
                                                config,
                                                boundary_vector_file=local_boundary_vector_file,
                                                boundary_upper_left=local_boundary_upper_left,
                                                b_set=boundary_sets[_site],
                                                boundary_subset_geotransform=subset_geotransform)

            if (local_feature is not None):
                features[sample_index, ...] = local_feature.copy()
                good_response_data[_cr] = True
                sample_index += 1
        responses_per_site[_site] = responses_per_site[_site][good_response_data, :]

    # transform responses
    responses = np.vstack(responses_per_site)
    del responses_per_site

    # Get the feature shapes for re-reading (modified ooc resize)
    feat_shape = list(features.shape)
    feat_shape[0] = sample_index

    # Delete and reload feauters, as a hard and fast way to force data dump to disc and reload
    # with a modified size....IE, an ooc resize
    del features
    features = np.memmap(feature_memmap_file, dtype=np.float32, mode='r+', shape=(tuple(feat_shape)))

    # Shuffle the data one last time (in case the fold-assignment would otherwise be biased beacuase of
    # the feature/response file order
    perm = np.random.permutation(features.shape[0])
    features = features[perm, :]
    responses = responses[perm, :]
    del perm

    fold_assignments = np.zeros(responses.shape[0]).astype(int)
    for f in range(0, config.n_folds):
        idx_start = int(round(f / config.n_folds * len(fold_assignments)))
        idx_finish = int(round((f + 1) / config.n_folds * len(fold_assignments)))
        fold_assignments[idx_start:idx_finish] = f

    weights = np.ones((responses.shape[0], 1))

    # one hot encode
    responses, response_band_types = one_hot_encode_array(response_raw_band_types, responses, response_memmap_file)

    _logger.info('Feature shape: {}'.format(features.shape))
    _logger.info('Response shape: {}'.format(responses.shape))
    _logger.info('Weight shape: {}'.format(weights.shape))

    for fold in range(config.n_folds):
        np.save(config.feature_files[fold], features[fold_assignments == fold, ...])
        np.save(config.response_files[fold], responses[fold_assignments == fold, ...])
        np.save(config.weight_files[fold], weights[fold_assignments == fold, ...])

    del features, responses, weights
    if ('C' in response_raw_band_types):
        if (np.sum(np.array(response_raw_band_types) == 'C') > 1):
            _logger.warning('Currently weighting is only enabled for one categorical response variable')
        features, responses, weights, success = open_memmap_files(config, writeable=True, override_success_file=True)
        weights = calculate_categorical_weights(responses, weights, config)
        del features, responses, weights

    # clean up munge files
    if (os.path.isfile(feature_memmap_file)):
        os.remove(feature_memmap_file)

    Path(config.successful_data_save_file).touch()
    features, responses, weights, success = open_memmap_files(config, writeable=False)
    return features, responses, weights, response_band_types


def _check_mask_data_sufficient(mask: np.array, max_nodata_fraction: float) -> bool:
    if mask is not None:
        nodata_fraction = np.sum(mask) / np.prod(mask.shape)
        if nodata_fraction <= max_nodata_fraction:
            _logger.trace('Data mask has sufficient data, missing data proportion: {}'.format(nodata_fraction))
            return True
        else:
            _logger.trace('Data mask has insufficient data, missing data proportion: {}'.format(nodata_fraction))
            return False
    else:
        _logger.trace('Data mask is None')
        return False
