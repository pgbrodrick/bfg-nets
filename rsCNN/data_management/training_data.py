import logging
import os
from typing import List, Tuple

import fiona
import gdal
import multiprocessing
import numpy as np
import ogr, osr
import rasterio.features
from tqdm import tqdm

from rsCNN.configuration import configs, sections
from rsCNN.data_management import common_io, ooc_functions, data_core
from rsCNN.utils import compute_access

_logger = logging.getLogger(__name__)

parallel_write_lock = multiprocessing.Lock()

def build_training_data_ordered(
        config: configs.Config,
        feature_raw_band_types: List[List[str]],
        response_raw_band_types: List[List[str]]
) -> Tuple[List[np.array], List[np.array], List[np.array], List[str], List[str]]:

    # TODO:  check default, and set it to a standard value
    if config.data_build.random_seed:
        _logger.debug('Setting random seed to {}'.format(config.data_build.random_seed))
        np.random.seed(config.data_build.random_seed)

    # TODO:  move to config checks
    if (isinstance(config.data_build.max_samples, list)):
        if (len(config.data_build.max_samples) != len(config.raw_files.feature_files)):
            raise Exception('max_samples must equal feature_files length, or be an integer.')

    n_features = int(np.sum([len(feat_type) for feat_type in feature_raw_band_types]))
    n_responses = int(np.sum([len(resp_type) for resp_type in response_raw_band_types]))

    # TODO:  move to checks
    feature_memmap_size_gb = n_features*4*config.data_build.max_samples * \
        (config.data_build.window_radius*2)**2 / 1024.**3
    assert feature_memmap_size_gb < config.data_build.max_built_data_gb,\
           'Expected feature memmap size: {} Gb, limit: {}'.format(feature_memmap_size_gb , config.data_build.max_built_data_gb)

    response_memmap_size_gb = n_responses*4*config.data_build.max_samples * \
        (config.data_build.window_radius*2)**2 / 1024.**3
    assert response_memmap_size_gb < config.data_build.max_built_data_gb,\
           'Expected feature memmap size: {} Gb, limit: {}'.format(response_memmap_size_gb , config.data_build.max_built_data_gb)

    features, responses = _open_temporary_features_responses_data_files(config, n_features, n_responses)
    _log_munged_data_information(features, responses)

    _logger.debug('Pre-compute all subset locations')
    all_site_upper_lefts = []
    all_site_xy_locations = []
    reference_subset_geotransforms = []
    for _site in range(0, len(config.raw_files.feature_files)):
        feature_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly)
                        for loc_file in config.raw_files.feature_files[_site]]
        response_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_files.response_files[_site]]
        boundary_set = common_io.get_site_boundary_set(config, _site)

        all_set_upper_lefts, xy_sample_locations = common_io.get_all_interior_extent_subset_pixel_locations(
            gdal_datasets=[feature_sets, response_sets, [bs for bs in [boundary_set] if bs is not None]],
            window_radius=config.data_build.window_radius,
            inner_window_radius=config.data_build.loss_window_radius,
            shuffle=True)

        all_site_upper_lefts.append(all_set_upper_lefts)
        all_site_xy_locations.append(xy_sample_locations)

        ref_trans = feature_sets[0].GetGeoTransform()
        subset_geotransform = None
        if config.raw_files.boundary_files is not None:
            if config.raw_files.boundary_files[_site] is not None and \
                    _is_boundary_file_vectorized(config.raw_files.boundary_files[_site]):
                subset_geotransform = [ref_trans[0], ref_trans[1], 0, ref_trans[3], 0, ref_trans[5]]
        reference_subset_geotransforms.append(subset_geotransform)

    num_reads_per_site = int(np.floor(config.data_build.max_samples / len(config.raw_files.feature_files)))

    _logger.debug('Step through sites in order and grab {} sample from each until max samples'.format(num_reads_per_site))
    progress_bar = tqdm(total=config.data_build.max_samples, ncols=80)


    write_lock = multiprocessing.Lock()
    n_cpus = compute_access.get_count_available_cpus()
    pool = multiprocessing.Pool(processes=n_cpus)
    read_results = []

    _sample_index = 0
    _rs_index = 0
    remaining_sites = list(range(len(config.raw_files.feature_files)))
    _site_xy_index = np.zeros(len(remaining_sites)).astype(int).tolist()
    all_strings = []
    index_success = np.zeros(config.data_build.max_samples)
    while (_sample_index < config.data_build.max_samples and len(remaining_sites) > 0):
        _site = remaining_sites[_rs_index]
        _logger.debug('Reading loop: Site {}'.format(_site))

        while _site_xy_index[_site] < len(all_site_xy_locations[_site]):
            site_read_count = 0
            _logger.debug('Site index: {}'.format(_site_xy_index[_site]))

            [f_ul, r_ul, [b_ul]] = all_site_upper_lefts[_site]

            local_boundary_vector_file = None
            local_boundary_upper_left = None
            if (b_ul is not None):
                local_boundary_upper_left = b_ul + all_site_xy_locations[_site][_site_xy_index[_site], :]
            subset_geotransform = None
            if (reference_subset_geotransforms[_site] is not None):
                ref_trans = reference_subset_geotransforms[_site]
                subset_geotransform[0] = ref_trans[0] + \
                    (f_ul[0][0] + all_site_xy_locations[_site_xy_index[_site], 0]) * ref_trans[1]
                subset_geotransform[3] = ref_trans[_site][3] + \
                    (f_ul[0][1] + all_site_xy_locations[_site_xy_index[_site], 1]) * ref_trans[5]
                local_boundary_vector_file = config.raw_files.boundary_files[_site]

            index_success[_sample_index] = -1
            read_results.append(pool.apply_async(read_segmentation_chunk, 
                                args=(\
                                _site,
                                f_ul + all_site_xy_locations[_site][_site_xy_index[_site], :],
                                r_ul + all_site_xy_locations[_site][_site_xy_index[_site], :],
                                config,
                                _sample_index,),
                                kwds=dict(boundary_vector_file=local_boundary_vector_file,
                                boundary_upper_left=local_boundary_upper_left,
                                boundary_subset_geotransform=subset_geotransform)))
            _sample_index = np.argmax(index_success == 0)

            _site_xy_index[_site] += 1
            popped = None
            if (_site_xy_index[_site] >= len(all_site_xy_locations[_site])):
                _logger.debug('All locations in site {} have been checked.'.format(
                    config.raw_files.feature_files[_site][0]))
                popped = remaining_sites.pop(_rs_index)


            if (len(read_results) > num_reads_per_site - site_read_count or \
                len(read_results) + np.sum(index_success == 1) > config.data_build.max_samples or popped is not None):
                read_output = [p.get() for p in read_results]
                pool.close()
                pool.join()

                for r in read_output:
                    print((r,np.sum(index_success == 1)))
                    if r[0] is True:
                        site_read_count += 1
                        progress_bar.update(1)
                        index_success[r[1]] = 1
                    else:
                        index_success[r[1]] = 0

                read_output = []

                if (popped is not None):
                    _rs_index += 1
                    if (_rs_index >= len(remaining_sites)):
                        _rs_index = 0

                if (site_read_count > num_reads_per_site or 
                    np.sum(index_success == 1) >= config.data_build.max_samples):
                    break
                else:
                    pool = multiprocessing.Pool(processes=n_cpus)

    progress_bar.close()
    del all_site_upper_lefts
    del all_site_xy_locations
    del reference_subset_geotransforms

    features, responses = _open_temporary_features_responses_data_files(config,
                                                                        n_features,
                                                                        n_responses,
                                                                        read_type='r+')
    features = _resize_munged_features(features, _sample_index, config)
    responses = _resize_munged_responses(responses, _sample_index, config)
    assert features.shape[0] > 0, 'Failed to collect a single sample'
    _log_munged_data_information(features, responses)

    _logger.debug('Shuffle data to avoid fold assignment biases')
    perm = np.random.permutation(features.shape[0])
    features = ooc_functions.permute_array(features, data_core.get_temporary_features_filepath(config), perm)
    responses = ooc_functions.permute_array(responses, data_core.get_temporary_responses_filepath(config), perm)
    del perm

    _logger.debug('Create uniform weights')
    shape = tuple(list(features.shape)[:-1] + [1])
    weights = np.memmap(data_core.get_temporary_weights_filepath(config), dtype=np.float32, mode='w+', shape=shape)
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
    features, feature_band_types = ooc_functions.one_hot_encode_array(
        feature_raw_band_types, features, data_core.get_temporary_features_filepath(config))
    _logger.debug('One-hot encode responses')
    responses, response_band_types = ooc_functions.one_hot_encode_array(
        response_raw_band_types, responses, data_core.get_temporary_responses_filepath(config))
    _log_munged_data_information(features, responses, weights)

    _save_built_data_files(features, responses, weights, config)
    del features, responses, weights

    if ('C' in response_raw_band_types):
        assert np.sum(np.array(response_raw_band_types) == 'C') == 1, \
            'Weighting is currently only enabled for one categorical response variable.'
        features, responses, weights = load_built_data_files(config, writeable=True)
        weights = calculate_categorical_weights(responses, weights, config)
        _logger.debug('Delete in order to flush output')
        del features, responses, weights

    _remove_temporary_data_files(config)

    _logger.debug('Store data build config sections')
    _save_built_data_config_sections_to_verify_successful(config)
    features, responses, weights = load_built_data_files(config, writeable=False)
    return features, responses, weights, feature_band_types, response_band_types


def build_training_data_from_response_points(
        config: configs.Config,
        feature_raw_band_types: List[List[str]],
        response_raw_band_types: List[List[str]]
) -> Tuple[List[np.array], List[np.array], List[np.array], List[str], List[str]]:

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
        boundary_set = common_io.get_site_boundary_set(config, _site)

        _logger.debug('Calculate overlapping extent')
        [f_ul, r_ul, [b_ul]], x_px_size, y_px_size = common_io.get_overlapping_extent(
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
        data_core.get_temporary_features_filepath(config), dtype=np.float32, mode='w+',
        shape=(total_samples, 2*config.data_build.window_radius, 2*config.data_build.window_radius, num_features)
    )

    sample_index = 0
    for _site in range(0, len(config.raw_files.feature_files)):
        _logger.debug('Open feature and response datasets for site {}'.format(_site))
        feature_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly)
                        for loc_file in config.raw_files.feature_files[_site]]
        response_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_files.response_files[_site]]
        boundary_set = common_io.get_site_boundary_set(config, _site)

        _logger.debug('Calculate interior rectangle location and extent')
        [f_ul, r_ul, [b_ul]], x_px_size, y_px_size = common_io.get_overlapping_extent(
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
    features = ooc_functions.permute_array(features, data_core.get_temporary_features_filepath(config), perm)
    responses = ooc_functions.permute_array(features, data_core.get_temporary_responses_filepath(config), perm)
    del perm

    weights = np.ones((responses.shape[0], 1))
    _log_munged_data_information(features, responses, weights)

    # one hot encode
    features, feature_band_types = ooc_functions.one_hot_encode_array(
        feature_raw_band_types, features, data_core.get_temporary_features_filepath(config))
    responses, response_band_types = ooc_functions.one_hot_encode_array(
        response_raw_band_types, responses, data_core.get_temporary_responses_filepath(config))
    _log_munged_data_information(features, responses, weights)

    # This change happens to work in this instance, but will not work with all sampling strategies.  I will leave for
    # now and refactor down the line as necessary.  Generally speaking, fold assignments are specific to the style of data read
    _save_built_data_files(features, responses, weights, config)
    del features, responses, weights

    if 'C' in response_raw_band_types:
        assert np.sum(np.array(response_raw_band_types) == 'C') == 1, \
            'Weighting is currently only enabled for one categorical response variable.'
        features, responses, weights = load_built_data_files(config, writeable=True)
        weights = calculate_categorical_weights(responses, weights, config)
        del features, responses, weights

    _remove_temporary_data_files(config)

    _save_built_data_config_sections_to_verify_successful(config)
    features, responses, weights = load_built_data_files(config, writeable=False)
    return features, responses, weights, feature_band_types, response_band_types


def rasterize_vector(vector_file: str, geotransform: List[int], output_shape: Tuple) -> np.array:
    """ Rasterizes an input vector directly into a numpy array.
    Arguments:
    vector_file
      Input vector file to be rasterized.
    geotransform
      A gdal style geotransform.
    output_shape
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


def get_proj(fname: str) -> str:
    """ Get the projection of a raster/vector dataset.
    Arguments:
    fname - str
      Name of input file.

    Returns:
    The projection of the input fname
    """
    ds = gdal.Open(fname, gdal.GA_ReadOnly)
    if (ds is not None):
        proj = ds.GetProjection()
        if (proj is not None):
            srs = osr.SpatialReference(wkt=str(proj))
            if srs.IsProjected is False:
                exception_str = 'File {} is not projected.  Correct or set ignore_projections flag.'.format(fname)
                raise Exception(exception_str)
            return srs.GetAttrValue('projcs')

    if (os.path.basename(fname).split('.')[-1] == 'shp'):
        vset = ogr.GetDriverByName('ESRI Shapefile').Open(fname, gdal.GA_ReadOnly)
    elif (os.path.basename(fname).split('.')[-1] == 'kml'):
        vset = ogr.GetDriverByName('KML').Open(fname, gdal.GA_ReadOnly)
    else:
        exception_str = 'Cannot find projection from file {}'.format(fname)
        raise Exception(exception_str)

    if (vset is None):
        exception_str = 'Cannot find projection from file {}'.format(fname)
        raise Exception(exception_str)
    else:
        proj = vset.GetLayer().GetSpatialRef()
        if (proj is None):
            exception_str = 'Cannot find projection from file {}'.format(fname)
            raise Exception(exception_str)
        else:
            srs = osr.SpatialReference(wkt=str(proj))
            if srs.IsProjected is False:
                exception_str = 'File {} is not projected.  Correct or set ignore_projections flag.'.format(fname)
                raise Exception(exception_str)
            return srs.GetAttrValue('projcs')



def check_projections(f_files: List[List[str]], r_files: List[List[str]], b_files: List[str] = None) -> List[str]:

    # f = feature, r = response, b = boundary
    errors = []

    if (b_files is None):
        b_files = []

    site_f_proj = []
    site_r_proj = []
    site_b_proj = []
    for _site in range(len(f_files)):

        f_proj = []
        r_proj = []
        for _file in range(len(f_files[_site])):
            f_proj.append(get_proj(f_files[_site][_file]))

        for _file in range(len(r_files[_site])):
            r_proj.append(get_proj(r_files[_site][_file]))

        b_proj = None
        if (len(b_files) > 0):
            b_proj = get_proj(b_files[_site])

        un_f_proj = np.unique(f_proj)
        if (len(un_f_proj) > 1):
            errors.append('Feature projection mismatch at site {}, projections: {}'.format(_site, un_f_proj))

        un_r_proj = np.unique(r_proj)
        if (len(un_r_proj) > 1):
            errors.append('Response projection mismatch at site {}, projections: {}'.format(_site, un_r_proj))

        if (un_f_proj[0] != un_r_proj[0]):
            errors.append('Feature/Response projection mismatch at site {}\nFeature proj: {}\nResponse proj: {}'.format(_site,un_f_proj[0],un_r_proj[0]))

        if (b_proj is not None):
            if (un_f_proj[0] != b_proj):
                errors.append('Feature/Boundary projection mismatch at site {}\nFeature proj: {}\nBoundary proj: {}'.format(_site,un_f_proj[0],b_proj))
            if (un_r_proj[0] != b_proj):
                errors.append('Response/Boundary projection mismatch at site {}\nFeature proj: {}\nBoundary proj: {}'.format(_site,un_r_proj[0],b_proj))

        site_f_proj.append(un_f_proj[0])
        site_r_proj.append(un_r_proj[0])
        if (b_proj is not None):
            site_b_proj.append(b_proj)

    if (len(np.unique(site_f_proj)) > 1):
        errors.append('Warning, different projections at different features sites: {}'.format(site_f_proj))
    if (len(np.unique(site_r_proj)) > 1):
        errors.append('Warning, different projections at different features sites: {}'.format(site_r_proj))
    if (len(np.unique(site_b_proj)) > 1):
        errors.append('Warning, different projections at different features sites: {}'.format(site_b_proj))
    return errors


def check_resolutions(f_files: List[List[str]], r_files: List[List[str]], b_files: List[str] = None) -> List[str]:
    # f = feature, r = response, b = boundary
    errors = []

    if (b_files is None):
        b_files = []

    site_f_res = []
    site_r_res = []
    site_b_res = []
    for _site in range(len(f_files)):

        f_res = []
        r_res = []
        for _file in range(len(f_files[_site])):
            f_res.append(np.array(gdal.Open(f_files[_site][_file], gdal.GA_ReadOnly).GetGeoTransform())[[1, 5]])

        for _file in range(len(r_files[_site])):
            if (r_files[_site][_file].split('.')[-1] not in sections.VECTORIZED_FILENAMES):
                r_res.append(np.array(gdal.Open(r_files[_site][_file], gdal.GA_ReadOnly).GetGeoTransform())[[1, 5]])

        b_res = None
        if (len(b_files) > 0):
            if (b_files[_site].split('.')[-1] not in sections.VECTORIZED_FILENAMES):
                b_res = np.array(gdal.Open(b_files[_site], gdal.GA_ReadOnly).GetGeoTransform())[[1, 5]]

        un_f_res = []
        if (len(f_res) > 0):
            f_res = np.vstack(f_res)
            un_f_res = [np.unique(f_res[:,0]), np.unique(f_res[:,1])]
            if (len(un_f_res[0]) > 1 or len(un_f_res[1]) > 1):
                errors.append('Feature resolution mismatch at site {}, resolutions: {}'.format(_site, un_f_res))

        un_r_res = []
        if (len(r_res) > 0):
            r_res = np.vstack(r_res)
            un_r_res = [np.unique(r_res[:,0]), np.unique(r_res[:,1])]
            if (len(un_r_res[0]) > 1 or len(un_r_res[1]) > 1):
                errors.append('Response resolution mismatch at site {}, resolutions: {}'.format(_site, un_r_res))

        if (b_res is not None):
            if (un_f_res[0][0] != b_res[0] or un_f_res[1][0] != b_res[1]):
                errors.append('Feature/Boundary resolution mismatch at site {}'.format(_site))
            if (un_r_res[0][0] != b_res[0] or un_r_res[1][0] != b_res[1]):
                errors.append('Response/Boundary resolution mismatch at site {}'.format(_site))

        site_f_res.append(un_f_res[0])
        if (len(un_r_res) > 0):
            site_r_res.append(un_r_res[0])
        if (b_res is not None):
            site_b_res.append(b_res)

    if (len(np.unique(site_f_res)) > 1):
        _logger.info('Warning, different resolutions at different features sites: {}'.format(site_f_res))
    if (len(np.unique(site_r_res)) > 1):
        _logger.info('Warning, different resolutions at different features sites: {}'.format(site_r_res))
    if (len(np.unique(site_b_res)) > 1):
        _logger.info('Warning, different resolutions at different features sites: {}'.format(site_b_res))
    return errors



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
                        boundary_upper_left: List[int] = None) -> np.array:

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


def read_segmentation_chunk(_site: int,
                            feature_upper_lefts: List[List[int]],
                            response_upper_lefts: List[List[int]],
                            config: configs.Config,
                            sample_index: int,
                            boundary_vector_file: str = None,
                            boundary_subset_geotransform: tuple = None,
                            boundary_upper_left: List[int] = None) -> bool:

    f_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly)
                    for loc_file in config.raw_files.feature_files[_site]]
    r_sets = [gdal.Open(loc_file, gdal.GA_ReadOnly) for loc_file in config.raw_files.response_files[_site]]
    b_set = common_io.get_site_boundary_set(config, _site)

    window_diameter = config.data_build.window_radius * 2

    mask = read_mask_chunk(boundary_vector_file,
                           boundary_subset_geotransform,
                           b_set,
                           boundary_upper_left,
                           window_diameter,
                           config.raw_files.boundary_bad_value)

    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        return False, sample_index

    local_response, mask = common_io.read_map_subset(r_sets, response_upper_lefts,
                                                     window_diameter, mask, config.raw_files.response_nodata_value)
    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        return False, sample_index

    if (config.data_build.response_min_value is not None):
        with np.errstate(invalid='ignore'):
            local_response[local_response < config.data_build.response_min_value] = np.nan
    if (config.data_build.response_max_value is not None):
        with np.errstate(invalid='ignore'):
            local_response[local_response > config.data_build.response_max_value] = np.nan
    mask[np.any(np.isnan(local_response), axis=-1)] = True

    if (mask is None):
        return False, sample_index
    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        return False, sample_index

    local_feature, mask = common_io.read_map_subset(f_sets, feature_upper_lefts,
                                                    window_diameter, mask, config.raw_files.feature_nodata_value)

    if not _check_mask_data_sufficient(mask, config.data_build.feature_nodata_maximum_fraction):
        return False, sample_index

    # Final check (propogate mask forward), and return
    parallel_write_lock.acquire()
    local_feature[mask, :] = np.nan
    local_response[mask, :] = np.nan

    features, responses = _open_temporary_features_responses_data_files(
        config, local_feature.shape[-1], local_response.shape[-1], read_type='r+')
    features[sample_index, ...] = local_feature
    responses[sample_index, ...] = local_response
    del features, responses
    parallel_write_lock.release()

    return True, sample_index


################### File/Dataset Opening Functions ##############################


def _open_temporary_features_responses_data_files(config: configs.Config, num_features: int, num_responses: int,
                                                  read_type: str = 'w+') -> Tuple[np.array, np.array]:
    basename = data_core.get_memmap_basename(config)
    shape = [config.data_build.max_samples, config.data_build.window_radius * 2, config.data_build.window_radius * 2]
    shape_features = tuple(shape + [num_features])
    shape_responses = tuple(shape + [num_responses])

    features_filepath = data_core.get_temporary_features_filepath(config)
    responses_filepath = data_core.get_temporary_responses_filepath(config)

    _logger.debug('Create temporary munged features data file with shape {} at {}'.format(
        shape_features, features_filepath))
    features = np.memmap(features_filepath, dtype=np.float32, mode=read_type, shape=shape_features)

    _logger.debug('Create temporary munged responses data file with shape {} at {}'.format(
        shape_responses, responses_filepath))
    responses = np.memmap(responses_filepath, dtype=np.float32, mode=read_type, shape=shape_responses)

    #features_dataset = h5py.File(features_filepath, read_type)
    #responses_dataset = h5py.File(responses_filepath, read_type)

    # if (read_type == 'w'):
    #    features = features_dataset.create_dataset('features', shape_features, dtype=np.float32, chunks=True)
    #    responses = responses_dataset.create_dataset('responses', shape_responses, dtype=np.float32, chunks=True)
    # else:
    #    features = features_dataset['features']
    #    responses = responses_dataset['responses']
    return features, responses


def _create_temporary_weights_data_files(config: configs.Config, num_samples: int) -> np.array:
    weights_filepath = data_core.get_temporary_weights_filepath(config)
    _logger.debug('Create temporary munged weights data file at {}'.format(weights_filepath))
    shape = tuple([num_samples, config.data_build.window_radius * 2, config.data_build.window_radius * 2, 1])
    return np.memmap(weights_filepath, dtype=np.float32, mode='w+', shape=shape)


def load_built_data_files(config: configs.Config, writeable: bool = False) \
        -> Tuple[List[np.array], List[np.array], List[np.array]]:
    _logger.debug('Loading built data files with writeable == {}'.format(writeable))
    feature_files = data_core.get_built_features_filepaths(config)
    response_files = data_core.get_built_responses_filepaths(config)
    weight_files = data_core.get_built_weights_filepaths(config)
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
    features_filepaths = data_core.get_built_features_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug('Save features fold {}'.format(idx_fold))
        np.save(features_filepaths[idx_fold], features_munged[fold_assignments == idx_fold, ...])

    _logger.debug('Save responses to memmapped arrays separated by folds')
    responses_filepaths = data_core.get_built_responses_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug('Save responses fold {}'.format(idx_fold))
        np.save(responses_filepaths[idx_fold], responses_munged[fold_assignments == idx_fold, ...])

    _logger.debug('Save weights to memmapped arrays separated by folds')
    weights_filepaths = data_core.get_built_weights_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug('Save weights fold {}'.format(idx_fold))
        np.save(weights_filepaths[idx_fold], weights_munged[fold_assignments == idx_fold, ...])


def _save_built_data_config_sections_to_verify_successful(config: configs.Config) -> None:
    filepath = data_core.get_built_data_config_filepath(config)
    _logger.debug('Saving built data config sections to {}'.format(filepath))
    configs.save_config_to_file(config, filepath, include_sections=['raw_files', 'data_build'])


def _remove_temporary_data_files(config: configs.Config) -> None:
    _logger.debug('Remove temporary munge files')
    if os.path.exists(data_core.get_temporary_features_filepath(config)):
        os.remove(data_core.get_temporary_features_filepath(config))
    if os.path.exists(data_core.get_temporary_responses_filepath(config)):
        os.remove(data_core.get_temporary_responses_filepath(config))
    if os.path.exists(data_core.get_temporary_weights_filepath(config)):
        os.remove(data_core.get_temporary_weights_filepath(config))


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
        data_core.get_temporary_features_filepath(config), dtype=np.float32, mode='r+', shape=new_features_shape)
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
        data_core.get_temporary_responses_filepath(config), dtype=np.float32, mode='r+', shape=new_responses_shape)
    return responses_munged


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
    return str(os.path.splitext(boundary_filepath)).lower() in sections.VECTORIZED_FILENAMES


def check_built_data_files_exist(config: configs.Config) -> bool:
    filepaths = \
        data_core.get_built_features_filepaths(config) + \
        data_core.get_built_responses_filepaths(config) + \
        data_core.get_built_weights_filepaths(config)
    missing_files = [filepath for filepath in filepaths if not os.path.exists(filepath)]
    if not missing_files:
        _logger.debug('Built data files found at paths: {}'.format(', '.join(filepaths)))
    else:
        _logger.warning('Built data files were not found at paths: {}'.format(', '.join(missing_files)))
    return not missing_files


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
