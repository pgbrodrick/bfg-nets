import os
from pathlib import Path
import re
from typing import List

import fiona
import gdal
import numpy as np
import numpy.matlib
import rasterio.features
from tqdm import tqdm

from rsCNN.utils import logging
from rsCNN.utils.general import *
from rsCNN.data_management import scalers, DataConfig

_logger = logging.get_child_logger(__name__)


_FILENAME_RESPONSES_SUFFIX = 'responses_{}.npy'
_FILENAME_FEATURES_SUFFIX = 'features_{}.npy'
_FILENAME_WEIGHTS_SUFFIX = 'weights_{}.npy'
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


_logger = logging.get_child_logger(__name__)


def build_or_load_rawfile_data(config, rebuild=False):

    data_container = Dataset()
    check_input_files(config.raw_feature_file_list, config.raw_response_file_list, config.boundary_file_list)

    data_container.feature_raw_band_types, = get_band_types(config.raw_feature_file_list, config.feature_raw_band_type_input)
    data_container.response_raw_band_types, = get_band_types(config.raw_response_file_list, config.response_raw_band_type_input)

    if (rebuild is False):
        features, responses, weights, read_success = open_memmap_files(config)

    if (read_success is False or rebuild is True):
            assert config.raw_feature_file_list is not [], 'feature files to pull data from are required'
            assert config.raw_response_file_list is not [], 'response files to pull data from are required'

            if (config.ignore_projections is False): 
                check_projections(config.raw_feature_file_list, 
                                  config.raw_response_file_list, 
                                  config.boundary_file_list)

            if (config.response_data_format == 'FCN'):
                features, responses, weights = build_training_data_ordered(config, 
                                                                           data_container.feature_raw_band_types, 
                                                                           data_container.response_raw_band_types)
            elif (config.response_data_format == 'CNN'):
                features, responses, weights = build_training_data_from_response_points(config, 
                                                                                        data_container.feature_raw_band_types, 
                                                                                        data_container.response_raw_band_types)
            else:
                raise NotImplementedError('Unknown response data format')

    data_container.features = features
    data_container.responses = responses
    data_container.weights = weights

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
    feature_files = get_built_features_filenames(config.data_save_name, config.n_folds)
    response_files = get_built_responses_filenames(config.data_save_name, config.n_folds)
    weight_files = get_built_weights_filenames(config.data_save_name, config.n_folds)
    # TODO:  Phil:  I get a failed read on the first file when running this and building... but I think everything works
    #  after that? I don't know if "failed read" is the right message for something that succeeds?
    # FABINA - if you get a failed read message, than you should also only be returning None...is that not the case?
    for fold in range(config.n_folds):
        if (os.path.isfile(feature_files[fold])):
            features.append(np.load(feature_files[fold], mmap_mode=mode))
        else:
            success = False
            _logger.debug('failed read at {}'.format(feature_files[fold]))
            break
        if (os.path.isfile(response_files[fold])):
            responses.append(np.load(response_files[fold], mmap_mode=mode))
        else:
            _logger.debug('failed read at {}'.format(response_files[fold]))
            success = False
            break
        if (os.path.isfile(weight_files[fold])):
            weights.append(np.load(weight_files[fold], mmap_mode=mode))
        else:
            _logger.debug('failed read at {}'.format(weight_files[fold]))
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
        raise Exception('Cannot find projection from file {}'.format(fname)

    if (vset is None):
        raise Exception('Cannot find projection from file {}'.format(fname)
    else:
        b_proj = vset.GetLayer().GetSpatialRef()
        if (b_proj is None):
            raise Exception('Cannot find projection from file {}'.format(fname)
        else:
            b_proj = re.sub('\W', '', str(b_proj))
    
    return b_proj


def check_projections(a_files, b_files, c_files=[]):
    
    loc_a_files = [item for sublist in l for item in a_files]
    loc_b_files = [item for sublist in l for item in b_files]
    if ( len(c_files) > 0):
        loc_c_files = [item for sublist in l for item in a_files]
    else:
        loc_c_files = []

    a_proj = []
    b_proj = []
    c_proj = []

    if (_f in range(len(a_files))):
        a_proj.append(get_proj(loc_a_files[_f]))
        b_proj.append(get_proj(loc_b_files[_f]))
        if (len(loc_c_files) > 0):
            c_proj.append(get_proj(loc_c_files[_f]))
    
    for _p in range(len(a_proj)):
        if (len(c_proj) > 0):
            assert (a_proj[_p] != b_proj[_p] and a_proj != c_proj[_p]), 'Projection_mismatch between {}: {}, {}: {}, {}: {}'.
                   format(a_proj[_p], loc_a_files[_p], b_proj[_p], loc_b_files[_p], c_proj[_p], loc_c_files[_p])
        else:
            assert (a_proj[_p] != b_proj[_p]), 'Projection_mismatch between {}: {}, {}: {}'.
                   format(a_proj[_p], loc_a_files[_p], b_proj[_p], loc_b_files[_p])
                
            
    

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
    if (lb == 0):
        ub = responses[0].shape[1]
    else:
        ub = -lb

    # get response/total counts (batch-wise)
    response_counts = np.zeros(responses[0].shape[-1])
    total_valid_count = 0
    for _array in range(len(responses)):
        if (_array is not config.validation_fold and _array is not config.test_fold):
            for ind in range(0, responses[_array].shape[0], batch_size):
                lr = (responses[_array])[ind:ind+batch_size, lb:ub, lb:ub, :]
                lr[lr == config.response_nodata_value] = np.nan
                total_valid_count += np.sum(np.isfinite(lr))
                for _r in range(0, len(response_counts)):
                    response_counts[_r] += np.nansum(lr[..., _r] == 1)

    # assign_weights
    for _array in range(len(responses)):
        for ind in range(0, responses[_array].shape[0], batch_size):

            lr = (responses[_array])[ind:ind+batch_size, ...]
            lw = np.zeros((lr.shape[0], lr.shape[1], lr.shape[2]))
            for _r in range(0, len(response_counts)):
                lw[lr[..., _r] == 1] = total_valid_count / response_counts[_r]

            lw[:, :lb, :] = 0
            lw[:, ub:, :] = 0
            lw[:, :, :lb] = 0
            lw[:, :, ub:] = 0

            weights[_array][ind:ind+batch_size, :, :, 0] = lw

    return weights


def get_data_section(ds, x, y, x_size, y_size):
    dat = np.zeros((x_size, y_size, ds.RasterCount))
    for _b in range(ds.RasterCount):
        dat[..., _b] = ds.GetRasterBand(_b+1).ReadAsArray(x, y, x_size, y_size)
    return dat


def get_response_data_section(ds, x, y, x_size, y_size, config):
    dat = get_data_section(ds, x, y, x_size, y_size)
    dat = np.squeeze(dat)
    if (config.response_min_value is not None):
        dat[dat < config.response_min_value] = config.response_nodata_value
    if (config.response_max_value is not None):
        dat[dat > config.response_max_value] = config.response_nodata_value
    return dat


def read_segmentation_chunk(f_set,
                            r_set,
                            feature_upper_left,
                            response_upper_left,
                            config,
                            boundary_vector_file=None,
                            boundary_subset_geotransform=None,
                            b_set=None,
                            boundary_upper_left=None):
    window_diameter = config.window_radius * 2

    # Start by checking if we're inside boundary, if there is one
    mask = None
    if (boundary_vector_file is not None):
        mask = rasterize_vector(boundary_vector_file, boundary_subset_geotransform, (window_diameter, window_diameter))
    if (b_set is not None):
        mask = b_set.ReadAsArray(boundary_upper_left[0], boundary_upper_left[1], window_diameter, window_diameter)

    if mask is None:
        mask = np.zeros((window_diameter, window_diameter)).astype(bool)
    else:
        mask = mask == config.boundary_bad_value
        if np.all(mask):
            return None, None

    # Next check to see if we have a response, if so read all
    local_response = np.zeros((window_diameter, window_diameter, r_set.RasterCount))
    for _b in range(r_set.RasterCount):
        local_response[:, :, _b] = r_set.GetRasterBand(
            _b+1).ReadAsArray(response_upper_left[0], response_upper_left[1], window_diameter, window_diameter)
    local_response[local_response == config.response_nodata_value] = np.nan
    # TODO:  isn't this just checking whether values are nan and then assigning them to nan again?
    # FABINA - no, it's also checking for other non-nan non-finite numpy values, which can (and sometimes are) read
    local_response[np.isfinite(local_response) is False] = np.nan
    local_response[mask, :] = np.nan

    if (config.response_min_value is not None):
        local_response[local_response < config.response_min_value] = np.nan
    if (config.response_max_value is not None):
        local_response[local_response > config.response_max_value] = np.nan

    if (np.all(np.isnan(local_response))):
        return None, None
    mask[np.any(np.isnan(local_response), axis=-1)] = True

    # Last, read in features
    local_feature = np.zeros((window_diameter, window_diameter, f_set.RasterCount))
    for _b in range(0, f_set.RasterCount):
        local_feature[:, :, _b] = f_set.GetRasterBand(
            _b+1).ReadAsArray(feature_upper_left[0], feature_upper_left[1], window_diameter, window_diameter)

    local_feature[local_feature == config.feature_nodata_value] = np.nan
    local_feature[np.isfinite(local_feature) is False] = np.nan
    local_feature[mask, :] = np.nan
    if (np.all(np.isnan(local_feature))):
        return None, None

    feature_nodata_fraction = np.sum(np.isnan(local_feature)) / np.prod(local_feature.shape)
    # TODO:  as implemented, this only checks the feature nodata fraction, do we want to do responses too?
    if feature_nodata_fraction > config.nodata_maximum_fraction:
        return None, None
    mask[np.any(np.isnan(local_response), axis=-1)] = True

    # Final check, and return
    # TODO:  haven't we already done these operations above?
    # FABINA - no, since the mask has built as it's gone forward
    local_feature[mask, :] = np.nan
    local_response[mask, :] = np.nan

    return local_feature, local_response



class Dataset:

    """ A container class that holds all sorts of data objects
    """
    def __init__(config: DataConfig):
        features = []
        responses = []
        weights = []

        feature_scalers = []
        response_scalers = []

        self.train_folds = None

        self.config = config


    def build_or_load_scalers(data_container, rebuild=False):

        data_config = data_container.config

        feat_scaler_atr = {'savename_base': data_config.data_save_name + '_feature_scaler'}
        feature_scaler = scalers.get_scaler(data_config.feature_scaler_name, feat_scaler_atr)
        resp_scaler_atr = {'savename_base': data_config.data_save_name + '_response_scaler'}
        response_scaler = scalers.get_scaler(data_config.response_scaler_name, resp_scaler_atr)
        feature_scaler.load()
        response_scaler.load()

        self.train_folds = [x for x in range(config.n_folds) if x is not config.validation_fold and x is not config.test_fold]
    
        if (feature_scaler.is_fitted is False or rebuild is True):
            # TODO: do better
            feature_scaler.fit(data_config.features[self.train_folds[0]])
            feature_scaler.save()
        if (response_scaler.is_fitted is False or rebuild is True):
            # TODO: do better
            response_scaler.fit(data_config.responses[self.train_folds[0]])
            response_scaler.save()

        data_container.feature_scaler = feature_scaler
        data_container.response_scaler = response_scaler


    def check_input_files(f_file_list, r_file_list, b_file_list):
        
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
                assert gdal.Open(f_file_list[_site][_band],gdal.GA_ReadOnly) is not None, 
                       'Could not open feature site {}, file {}'.format(_site,_band)
            for _band in range(len(r_file_list[_site])):
                assert gdal.Open(r_file_list[_site][_band],gdal.GA_ReadOnly) is not None, 
                       'Could not open response site {}, file {}'.format(_site,_band)

        # Checks on the number of files per site
        num_f_files_per_site = len(f_file_list[0])
        num_r_files_per_site = len(r_file_list[0])
        for _site in range(len(f_file_list)):
            assert len(f_file_list[_site]) == num_f_files_per_site, 'Inconsistent number of feature files at site {}'.format(_site)
            assert len(r_file_list[_site]) == num_r_files_per_site, 'Inconsistent number of response files at site {}'.format(_site)

        # Checks on the number of bands per file
        num_f_bands_per_file = [gdal.Open(x,gdal.GA_ReadOnly).RasterCount for x in f_file_list[0]]
        num_r_bands_per_file = [gdal.Open(x,gdal.GA_ReadOnly).RasterCount for x in r_file_list[0]]
        for _site in range(len(f_file_list)):
            for _file in range(len(f_file_list[_site])):
                assert gdal.Open(f_file_list[_site][_file],gdal.GA_ReadOnly).RasterCount == num_f_bands_per_file[_file],
                       'Inconsistent number of feature bands in site {}, file {}'.format(_site,_band)

            for _file in range(len(r_file_list[_site])):
                assert gdal.Open(r_file_list[_site][_file],gdal.GA_ReadOnly).RasterCount == num_r_bands_per_site[_file]
                       'Inconsistent number of response bands in site {}, file {}'.format(_site,_band)


    def get_band_types(file_list, band_types):

        valid_band_types = ['R','C']
        # 3 options are available for specifying band_types:
        # 1) band_types is None - assume all bands are real
        # 2) band_types is a list of strings within valid_band_types - assume each band from the associated file is the specified type,
        #    requires len(band_types) == len(file_list[0])
        # 3) band_types is list of lists (of strings, contained in valid_band_types), with the outer list referring to 
        #    files and the inner list referring to bands

        num_bands_per_file = [gdal.Open(x,gdal.GA_ReadOnly).RasterCount for x in file_list[0]]

        # Nonetype, option 1 from above, auto-generate
        if (band_types is None):
            for _file in range(len(file_list[0])):
                output_raw_band_types = []
                output_raw_band_types.append(['R' for _band in range(num_bands_per_file[_file])])

        else:
            assert(type(band_types) is list, 'band_types must be None or a list')

            # List of lists, option 3 from above - just check components
            if (type(band_types[0]) is list):
                for _file in range(len(band_types)):
                    assert(type(band_types[_file]) is list, 'If one element of band_types is a list, all elements must be lists'
                    assert len(band_types[_file]) == num_bands_per_file[_file], 'File {} has wrong number of band types'.format(_file)
                    for _band in range(len(band_types[_file])):
                        assert band_types[_file][_band] in valid_band_types, 'Invalid band types at file {}, band {}'.format(_file,_band)
                
                output_raw_band_types = band_types

            else:
                # List of values valid_band_types, option 2 from above - convert to list of lists
                output_raw_band_types = []
                for _file in range(len(band_types)):
                    assert band_types[_file] in valid_band_types, 'Invalid band type at File {}'.format(_file)
                    output_raw_band_types.append([band_types[_file] for _band in range(num_bands_per_file[_file])])
            
        return output_raw_band_types




def upper_left_pixel(trans, interior_x, interior_y):
    x_ul = max((trans[0] - interior_x)/trans[1], 0)
    y_ul = max((interior_y - trans[3])/trans[5], 0)
    return x_ul, y_ul


def get_interior_rectangle(feature_set, response_set, boundary_set):
    f_trans = feature_set.GetGeoTransform()
    r_trans = response_set.GetGeoTransform()
    b_trans = None
    if (boundary_set is not None):
        b_trans = boundary_set.GetGeoTransform()

    # Calculate the interior space location and extent

    # Find the interior (UL) x,y coordinates in map-space
    interior_x = max(r_trans[0], f_trans[0])
    interior_y = min(r_trans[3], f_trans[3])
    if (b_trans is not None):
        interior_x = max(interior_x, b_trans[0])
        interior_y = max(interior_y, b_trans[3])

    # calculate the feature and response UL coordinates in pixel-space
    f_x_ul, f_y_ul = upper_left_pixel(f_trans, interior_x, interior_y)
    r_x_ul, r_y_ul = upper_left_pixel(r_trans, interior_x, interior_y)

    # calculate the size of the matched interior extent
    x_len = min(feature_set.RasterXSize - f_x_ul, response_set.RasterXSize - r_x_ul)
    y_len = min(feature_set.RasterYSize - f_y_ul, response_set.RasterYSize - r_y_ul)

    # update the UL location, and the interior extent, if there is a boundary
    if (b_trans is not None):
        b_x_ul, b_y_ul = upper_left_pixel(b_trans, interior_x, interior_y)
        x_len = min(x_len, boundary_set.RasterXSize - b_x_ul)
        y_len = min(y_len, boundary_set.RasterYSize - b_y_ul)

    # convert these UL coordinates to an array for easy addition later
    f_ul = np.array([f_x_ul, f_y_ul])
    r_ul = np.array([r_x_ul, r_y_ul])
    if (b_trans is not None):
        b_ul = np.array([b_x_ul, b_y_ul])
    else:
        b_ul = None

    return f_ul, r_ul, b_ul, x_len, y_len




def build_training_data_ordered(config: DataConfig, raw_feature_band_types, raw_response_band_types):
    """ Builds a set of training data based on the configuration input
        Arguments:
        config - object of type Data_Config with the requisite values for preparing training data (see __init__.py)

    """
    
    if (config.random_seed is not None):
        np.random.seed(config.random_seed)

    if (isinstance(config.max_samples, list)):
        if (len(config.max_samples) != len(config.raw_feature_file_list)):
            raise Exception('max_samples must equal feature_file_list length, or be an integer.')

    feature_set = gdal.Open(config.raw_feature_file_list[0], gdal.GA_ReadOnly)
    response_set = gdal.Open(config.raw_response_file_list[0], gdal.GA_ReadOnly)
    n_features = feature_set.RasterCount

    feature_memmap_file = config.data_save_name + '_feature_munge_memmap.npy'
    response_memmap_file = config.data_save_name + '_response_munge_memmap.npy'
    weight_memmap_file = config.data_save_name + '_weight_munge_memmap.npy'

    # TODO: fix max size issue, but force for now to prevent overly sized sets
    assert(config.max_samples * (config.window_radius*2)**2 * n_features / 1024.**3 < 10, 'max_samples too large')
    features = np.memmap(feature_memmap_file,
                         dtype=np.float32,
                         mode='w+',
                         shape=(config.max_samples, config.window_radius*2, config.window_radius*2, n_features))

    responses = np.memmap(response_memmap_file,
                          dtype=np.float32,
                          mode='w+',
                          shape=(config.max_samples, config.window_radius*2, config.window_radius*2, response_set.RasterCount))

    sample_index = 0
    for _i in range(0, len(config.raw_feature_file_list)):

        # open requisite datasets
        feature_set = gdal.Open(config.raw_feature_file_list[_i], gdal.GA_ReadOnly)
        response_set = gdal.Open(config.raw_response_file_list[_i], gdal.GA_ReadOnly)
        boundary_set = None

        # this boolean could be simplified to two levels pretty easily, maybe even just one, but then it could
        # also be hidden in the above mentioned object
        # FABINA - Fair, I've simplified it to two lines.  I don't think it can safely go to one, could be wrong.
        if (len(config.boundary_file_list) > 0):
            if (config.boundary_file_list[_i] is not None and config.boundary_as_vectors[_i]):
                boundary_set = gdal.Open(config.boundary_file_list[_i], gdal.GA_ReadOnly)

        f_trans = feature_set.GetGeoTransform()
        r_trans = response_set.GetGeoTransform()
        b_trans = None
        if (boundary_set is not None):
            b_trans = boundary_set.GetGeoTransform()

        # Calculate the interior space location and extent
        f_ul, r_ul, b_ul, x_len, y_len = get_interior_rectangle(feature_set, response_set, boundary_set)

        collist = [x for x in range(0,
                                    int(x_len - 2*config.window_radius),
                                    int(config.internal_window_radius*2))]
        rowlist = [y for y in range(0,
                                    int(y_len - 2*config.window_radius),
                                    int(config.internal_window_radius*2))]

        colrow = np.zeros((len(collist)*len(rowlist), 2)).astype(int)
        colrow[:, 0] = np.matlib.repmat(np.array(collist).reshape((-1, 1)), 1, len(rowlist)).flatten()
        colrow[:, 1] = np.matlib.repmat(np.array(rowlist).reshape((1, -1)), len(collist), 1).flatten()
        # IF these operations happen in functions, you don't need to worry about manually deleting these objects
        # because they're lost when the function exits, i.e., they're only in the local scope
        # FABINA - While that is true, what we're doing here is trying to free back up memory, and deleting
        # thee objects does help with that.
        del collist, rowlist

        # There's a shuffle function that does this in np
        # FABINA - probably...is there a problem with this implementation (or is this hard to read???)
        colrow = colrow[np.random.permutation(colrow.shape[0]), :]

        subset_geotransform = None
        if (len(config.boundary_file_list) > 0):
            if config.boundary_file_list[_i] is not None and is_boundary_file_vectorized(config.boundary_file_list[_i]):
                subset_geotransform=[f_trans[0], f_trans[1], 0, f_trans[3], 0, f_trans[5]]

        # same comments about turning nested content into a function, turning math into informatively named
        # functions, etc
        # FABINA - I can maybe follow you with the above....but for the below, it's alreayd broken out into a
        # function....this loop is only assigning the appropriate option.  What would be a better way?
        # Update - I've condensed the code below a bit to see if that helps at all.  Functionally the same, but
        # not a bit tighter to read
        for _cr in tqdm(range(len(colrow)), ncols=80):

            # Determine local information about boundary file
            local_boundary_vector_file = None
            local_boundary_upper_left = None
            if (boundary_set is not None):
                local_boundary_set = boundary_set
                local_boundary_upper_left = b_ul + colrow[_cr, :]
            if (subset_geotransform is not None):
                subset_geotransform[0] = f_trans[0] + (f_ul[0] + colrow[_cr, 0]) * f_trans[1]
                subset_geotransform[3] = f_trans[3] + (f_ul[1] + colrow[_cr, 1]) * f_trans[5]
                local_boundary_vector_file = config.boundary_file_list[_i]

            local_feature, local_response = read_segmentation_chunk(feature_set,
                                                                    response_set,
                                                                    f_ul + colrow[_cr, :],
                                                                    r_ul + colrow[_cr, :],
                                                                    config,
                                                                    boundary_vector_file=local_boundary_vector_file,
                                                                    boundary_upper_left=local_boundary_upper_left,
                                                                    b_set=boundary_set,
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

    # Honestly, at this point my head is starting to hurt. I'm trying to remember what has happened in this
    # function up until this point because I was trying to make a mental "table of contents", but I can't
    # do that without really focusing and looking back.
    # You can do this in one line with something like np.repeat, if this is doing what I think
    # FABINA -Maybe/probably you can swap this with a different set of lines that do basically the same thing.  But
    # this is trivial code to follow.....it's literally a division, and has appropriate names.  And is 3 lines.
    fold_assignments = np.zeros(responses.shape[0]).astype(int)
    for f in range(0, config.n_folds):
        idx_start = int(round(f / config.n_folds * len(fold_assignments)))
        idx_finish = int(round((f + 1) / config.n_folds * len(fold_assignments)))
        fold_assignments[idx_start:idx_finish] = f

    weights = np.memmap(weight_memmap_file,
                        dtype=np.float32,
                        mode='w+',
                        shape=(features.shape[0], features.shape[1], features.shape[2], 1))
    weights[:, :, :, :] = 1
    weights[np.isnan(responses[..., 0])] = 0

    # TODO: Why not just set all weights to zeros initially, then do the internal window set to ones, then do anything
    # isnan is zero? Would that be less code?
    # FABINA - I think it would be basically the same
    if (config.internal_window_radius != config.window_radius):
        buf = config.window_radius - config.internal_window_radius
        weights[:, :buf, :, -1] = 0
        weights[:, -buf:, :, -1] = 0
        weights[:, :, :buf, -1] = 0
        weights[:, :, -buf:, -1] = 0

    _logger.debug('Feature shape: {}'.format(features.shape))
    _logger.debug('Response shape: {}'.format(responses.shape))
    _logger.debug('Weight shape: {}'.format(weights.shape))

    #TODO - refactor based on the new info we have per band
    if any('C' in sublist for sublist in response_raw_band_types):
    if (config.data_build_category == 'ordered_categorical'):
        # TODO:  Maybe we need a check here that un_resp not too long?
        un_resp = np.unique(responses[np.isfinite(responses)])
        un_resp = un_resp[un_resp != config.response_nodata_value]
        _logger.debug('Found {} categorical responses'.format(len(un_resp)))

        resp_shape = list(responses.shape)
        resp_shape[-1] = len(un_resp)

        cat_response_memmap_file = config.data_save_name + '_cat_response_munge_memmap.npy'
        cat_responses = np.memmap(cat_response_memmap_file,
                                  dtype=np.float32,
                                  mode='w+',
                                  shape=tuple(resp_shape))

        # TODO:  Are you trying to iterate backwards? If so, it's clearer to write reversed(range(len(un_resp)))
        # Also, I wonder whether there's an off-by-one error here? Should that really be len(un_resp) - 1?
        # FABINA - yes, I was trying to iterate backwards, and no, there is no off-by-one erro, python being 0-based.
        # Doesn't need to be reversed now, as I swapped in from being in-place, so I've removed (doesn't matter, but
        # no need to confuse folks)

        # One hot-encode
        for _r in range(len(un_resp)):
            cat_responses[..., _r] = np.squeeze(responses[..., 0] == un_resp[_r])

        # Force file dump, and then reload the encoded responses as the primary response
        del responses, cat_responses
        os.remove(response_memmap_file)
        response_memmap_file=cat_response_memmap_file
        responses=np.memmap(response_memmap_file, dtype=np.float32, mode='r+', shape=tuple(resp_shape))

    feature_files = get_built_features_filenames(config.data_save_name, config.n_folds)
    response_files = get_built_responses_filenames(config.data_save_name, config.n_folds)
    weight_files = get_built_weights_filenames(config.data_save_name, config.n_folds)

    for fold in range(config.n_folds):
        np.save(feature_files[fold], features[fold_assignments == fold, ...])
        np.save(response_files[fold], responses[fold_assignments == fold, ...])
        np.save(weight_files[fold], weights[fold_assignments == fold, ...])

    del features, responses, weights
    if (config.data_build_category == 'ordered_categorical'):

        features, responses, weights, success = open_memmap_files(config, writeable=True, override_success_file=True)
        # TODO:  Phil:  it's currently possible for weights to be set to 0 in the entire loss window and then to have
        #   samples with no weights for the loss function. If a sample is composed of all overweighted classes, this
        #   could cause hard errors, but it seems like it's pretty suboptimal to waste any CPU/GPU cycles on samples
        #   with no weights, right?
        # FABINA - I don't understand how you could have an all-zero weight set, unless your max_nodata_fraction
        # is 100%.  Can you clarify?
        # Thinking through more, I see that this could occur if there are only values between the boundary and the
        # interior window.  This best handled by modifying the read-chunk code to use the max_nodata_fraction
        # within the interior window (for the responses), which was my original implemenation I believe (just didn't
        # get ported over).  Is this what you were thinking of?
        weights = calculate_categorical_weights(responses, weights, config)
        del features, responses, weights

    # clean up munge files
    if (os.path.isfile(feature_memmap_file)):
        os.remove(feature_memmap_file)
    if (os.path.isfile(response_memmap_file)):
        os.remove(response_memmap_file)
    if (os.path.isfile(weight_memmap_file)):
        os.remove(weight_memmap_file)

    Path(config.successful_data_save_file).touch()
    features, responses, weights, success = open_memmap_files(config, writeable=False)
    return features, responses, weights








def build_training_data_from_response_points(config: DataConfig, feature_raw_band_types, response_raw_band_types):
    """ Builds a set of training data based on the configuration input
        Arguments:
        config - object of type Data_Config with the requisite values for preparing training data (see __init__.py)

    """
    
    if (config.random_seed is not None):
        np.random.seed(config.random_seed)

    colrow_per_file = []
    responses_per_file = []
    for _i in range(0, len(config.raw_response_file_list)):
        # open requisite datasets
        feature_set = gdal.Open(config.raw_feature_file_list[_i], gdal.GA_ReadOnly)
        response_set = gdal.Open(config.raw_response_file_list[_i], gdal.GA_ReadOnly)
        boundary_set = gdal.Open(config.boundary_file_list[_i], gdal.GA_ReadOnly)

        # Calculate the interior space location and extent
        f_ul, r_ul, b_ul, x_len, y_len = get_interior_rectangle(feature_set, response_set, boundary_set)

        # Run through responses first
        collist = []
        rowlist = []
        responses = []
        for _line in range(y_len):
            line_dat = np.squeeze(response_set.ReadAsArray(r_ul[0], r_ul[1], x_len, 1).astype(np.float32))
            if (len (line_dat.shape) == 1):
                line_dat = line_dat.reshape(-1,1)

            if (config.response_background_value is not None):
                good_data = np.all(line_dat != config.response_background_value,axis=1)

            if (np.sum(good_data) > 0):
                line_x = np.arange(x_len)
                line_y = np.arange(y_len)

                collist.extend(line_x[good_data].tolist())
                rowlist.extend(line_y[good_data].tolist())

                responses.append(line_dat[good_data,:])

        colrow = np.vstack([np.array(collist),np.array(rowlist)]).T
        colrow_per_file.append(colrow)
        responses_per_file.apend(np.vstack(responses))

    max_samples = 0
    for _i in range(0, len(responses_per_file)):
        max_samples += responses_per_file[_i].shape[0]

    assert max_samples > 0, 'need more than 1 valid sample...'

    n_features = gdal.Open(raw_response_file_list[0],gdal.GA_ReadAsArray).RasterCount

    # TODO: fix max size issue, but force for now to prevent overly sized sets
    feature_memmap_file = config.data_save_name + '_feature_munge_memmap.npy'
    assert(max_samples * (config.window_radius*2)**2 * n_features / 1024.**3 < 10, 'max_samples too large')
    features = np.memmap(feature_memmap_file,
                         dtype=np.float32,
                         mode='w+',
                         shape=(config.max_samples, config.window_radius*2, config.window_radius*2, n_features))


    for _i in range(0, len(config.raw_response_file_list)):
        feature_set = gdal.Open(config.raw_feature_file_list[_i], gdal.GA_ReadOnly)
        response_set = gdal.Open(config.raw_response_file_list[_i], gdal.GA_ReadOnly)
        boundary_set = gdal.Open(config.boundary_file_list[_i], gdal.GA_ReadOnly)

        # Calculate the interior space location and extent
        f_ul, r_ul, b_ul, x_len, y_len = get_interior_rectangle(feature_set, response_set, boundary_set)

        colrow = colrow_per_file[_i]

        subset_geotransform = None
        if (len(config.boundary_file_list) > 0):
            if config.boundary_file_list[_i] is not None and is_boundary_file_vectorized(config.boundary_file_list[_i]):
                subset_geotransform=[f_trans[0], f_trans[1], 0, f_trans[3], 0, f_trans[5]]


        good_response_data = np.zeros(responses_per_file[_i].shape[0]).astype(bool)
        # Now read in features
        for _cr in tqdm(range(len(colrow)), ncols=80):

            # Determine local information about boundary file
            local_boundary_vector_file = None
            local_boundary_upper_left = None
            if (boundary_set is not None):
                local_boundary_set = boundary_set
                local_boundary_upper_left = b_ul + colrow[_cr, :]
            if (subset_geotransform is not None):
                subset_geotransform[0] = f_trans[0] + (f_ul[0] + colrow[_cr, 0]) * f_trans[1]
                subset_geotransform[3] = f_trans[3] + (f_ul[1] + colrow[_cr, 1]) * f_trans[5]
                local_boundary_vector_file = config.boundary_file_list[_i]

            local_feature = read_labeling_chunk(feature_set,
                                                f_ul + colrow[_cr, :],
                                                config,
                                                boundary_vector_file=local_boundary_vector_file,
                                                boundary_upper_left=local_boundary_upper_left,
                                                b_set=boundary_set,
                                                boundary_subset_geotransform=subset_geotransform)

            if (local_feature is not None):
                features[sample_index, ...] = local_feature.copy()
                good_response_data[_i] = True
                sample_index += 1
        responses_per_file[_i] = responses_per_file[_i][good_response_data,:]
    
    # transform responses
    responses = np.vstack(responses_per_file)
    del responses_per_file

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

    weights = np.ones(responses.shape[0])

    _logger.debug('Feature shape: {}'.format(features.shape))
    _logger.debug('Response shape: {}'.format(responses.shape))
    _logger.debug('Weight shape: {}'.format(weights.shape))


    
    #TODO - refactor based on the new info we have per band
    if (config.data_build_category == 'ordered_categorical'):
        # TODO:  Maybe we need a check here that un_resp not too long?
        un_resp = np.unique(responses[np.isfinite(responses)])
        un_resp = un_resp[un_resp != config.response_nodata_value]
        _logger.debug('Found {} categorical responses'.format(len(un_resp)))

        resp_shape = list(responses.shape)
        resp_shape[-1] = len(un_resp)

        cat_response_memmap_file = config.data_save_name + '_cat_response_munge_memmap.npy'
        cat_responses = np.memmap(cat_response_memmap_file,
                                  dtype=np.float32,
                                  mode='w+',
                                  shape=tuple(resp_shape))

        # TODO:  Are you trying to iterate backwards? If so, it's clearer to write reversed(range(len(un_resp)))
        # Also, I wonder whether there's an off-by-one error here? Should that really be len(un_resp) - 1?
        # FABINA - yes, I was trying to iterate backwards, and no, there is no off-by-one erro, python being 0-based.
        # Doesn't need to be reversed now, as I swapped in from being in-place, so I've removed (doesn't matter, but
        # no need to confuse folks)

        # One hot-encode
        for _r in range(len(un_resp)):
            cat_responses[..., _r] = np.squeeze(responses[..., 0] == un_resp[_r])

        # Force file dump, and then reload the encoded responses as the primary response
        del responses, cat_responses
        os.remove(response_memmap_file)
        response_memmap_file=cat_response_memmap_file
        responses=np.memmap(response_memmap_file, dtype=np.float32, mode='r+', shape=tuple(resp_shape))

    feature_files = get_built_features_filenames(config.data_save_name, config.n_folds)
    response_files = get_built_responses_filenames(config.data_save_name, config.n_folds)
    weight_files = get_built_weights_filenames(config.data_save_name, config.n_folds)

    for fold in range(config.n_folds):
        np.save(feature_files[fold], features[fold_assignments == fold, ...])
        np.save(response_files[fold], responses[fold_assignments == fold, ...])
        np.save(weight_files[fold], weights[fold_assignments == fold, ...])

    del features, responses, weights
    if (config.data_build_category == 'ordered_categorical'):

        features, responses, weights, success = open_memmap_files(config, writeable=True, override_success_file=True)
        # TODO:  Phil:  it's currently possible for weights to be set to 0 in the entire loss window and then to have
        #   samples with no weights for the loss function. If a sample is composed of all overweighted classes, this
        #   could cause hard errors, but it seems like it's pretty suboptimal to waste any CPU/GPU cycles on samples
        #   with no weights, right?
        # FABINA - I don't understand how you could have an all-zero weight set, unless your max_nodata_fraction
        # is 100%.  Can you clarify?
        # Thinking through more, I see that this could occur if there are only values between the boundary and the
        # interior window.  This best handled by modifying the read-chunk code to use the max_nodata_fraction
        # within the interior window (for the responses), which was my original implemenation I believe (just didn't
        # get ported over).  Is this what you were thinking of?
        weights = calculate_categorical_weights(responses, weights, config)
        del features, responses, weights

    # clean up munge files
    if (os.path.isfile(feature_memmap_file)):
        os.remove(feature_memmap_file)
    if (os.path.isfile(response_memmap_file)):
        os.remove(response_memmap_file)
    if (os.path.isfile(weight_memmap_file)):
        os.remove(weight_memmap_file)

    Path(config.successful_data_save_file).touch()
    features, responses, weights, success = open_memmap_files(config, writeable=False)
    return features, responses, weights



def get_built_features_filenames(data_save_name: str, num_folds: int) -> List[str]:
    basename = os.path.basename(data_save_name)
    if basename:
        data_save_name += '_'
    return [data_save_name + _FILENAME_FEATURES_SUFFIX.format(idx_fold) for idx_fold in range(num_folds)]


def get_built_responses_filenames(data_save_name: str, num_folds: int) -> List[str]:
    basename = os.path.basename(data_save_name)
    if basename:
        data_save_name += '_'
    return [data_save_name + _FILENAME_RESPONSES_SUFFIX.format(idx_fold) for idx_fold in range(num_folds)]


def get_built_weights_filenames(data_save_name: str, num_folds: int) -> List[str]:
    basename = os.path.basename(data_save_name)
    if basename:
        data_save_name += '_'
    return [data_save_name + _FILENAME_WEIGHTS_SUFFIX.format(idx_fold) for idx_fold in range(num_folds)]


def is_boundary_file_vectorized(boundary_filepath: str) -> bool:
    return os.path.splitext(boundary_filepath).lower() in _VECTORIZED_FILENAMES
