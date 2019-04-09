import gdal
import os
import re
from tqdm import tqdm

from rsCNN.utils import logger
from rsCNN.utils.general import *
from rsCNN.data_management import scalers


_logger = logger.get_child_logger(__name__)


def build_or_load_scalers(data_config, rebuild=False):
        feat_scaler_atr = {'nodata_value': data_config.feature_nodata_value,
                           'savename_base': data_config.data_save_name + '_feature_scaler'}
        feature_scaler = scalers.get_scaler(data_config.feature_scaler_name,
                                                 feat_scaler_atr)

        resp_scaler_atr = {'nodata_value': data_config.response_nodata_value,
                           'savename_base': data_config.data_save_name + '_response_scaler'}
        response_scaler = scalers.get_scaler(data_config.response_scaler_name,
                                                  resp_scaler_atr)
        feature_scaler.load()
        response_scaler.load()

        train_folds = [x for x in np.arange(
            data_config.n_folds) if x is not data_config.validation_fold and x is not data_config.test_fold]

        if (feature_scaler.is_fitted is False or rebuild is True):
            # TODO: do better
            feature_scaler.fit(data_config.features[train_folds[0]])
            feature_scaler.save()
        if (response_scaler.is_fitted is False or rebuild is True):
            # TODO: do better
            response_scaler.fit(data_config.responses[train_folds[0]])
            response_scaler.save()

        data_config.feature_scaler = feature_scaler
        data_config.response_scaler = feature_scaler


def build_or_load_data(config, rebuild=False):

    if (rebuild is False):
        features, responses, weights, read_success = load_training_data(config)

    if (read_success is False or rebuild is True):
        if (config.data_build_category in ['ordered_continuous', 'ordered_categorical']):
            features, responses, weights = build_training_data_ordered(config)
        else:
            raise NotImplementedError('Unknown data_build_category')

    config.features = features
    config.responses = responses
    config.weights = weights


def load_training_data(config):
    """
        Loads and returns training data from disk based on the config savename
        Arguments:
            config - data config from which to reference data
        Returns:
            features - feature data
            responses - response data
            fold_assignments - per-sample fold assignments specified during data generation
    """

    success = True
    if (os.path.isfile(config.successful_data_save_file) is not True):
        _logger.debug('no saved data')
        success = False

    features = []
    responses = []
    weights = []
    for fold in range(config.n_folds):
        if (os.path.isfile(config.feature_files[fold])):
            features.append(np.load(config.feature_files[fold], mmap_mode='r'))
        else:
            success = False
            _logger.debug('failed read at {}'.format(config.feature_files[fold]))
            break
        if (os.path.isfile(config.response_files[fold])):
            responses.append(np.load(config.response_files[fold], mmap_mode='r'))
        else:
            _logger.debug('failed read at {}'.format(config.response_files[fold]))
            success = False
            break
        if (os.path.isfile(config.weight_files[fold])):
            weights.append(np.load(config.weight_files[fold], mmap_mode='r'))
        else:
            _logger.debug('failed read at {}'.format(config.weight_files[fold]))
            success = False
            break

    if (success):
        return features, responses, weights, True
    else:
        return None, None, None, False


def get_proj(fname, is_vector):
    """ Get the projection of a raster/vector dataset.
    Arguments:
    fname - str
      Name of input file.
    is_vector - boolean
      Boolean indication of whether the file is a vector or a raster.

    Returns:
    The projection of the input fname
    """
    if (is_vector):
        if (os.path.basename(fname).split('.')[-1] == 'shp'):
            vset = ogr.GetDriverByName('ESRI Shapefile').Open(fname, gdal.GA_ReadOnly)
        elif (os.path.basename(fname).split('.')[-1] == 'kml'):
            vset = ogr.GetDriverByName('KML').Open(fname, gdal.GA_ReadOnly)
        else:
            raise Exception('unsupported vector file type from file ' + fname)

        b_proj = vset.GetLayer().GetSpatialRef()
    else:
        b_proj = gdal.Open(fname, gdal.GA_ReadOnly).GetProjection()

    return re.sub('\W', '', str(b_proj))


def check_data_matches(set_a, set_b, set_b_is_vector=False, set_c=[], set_c_is_vector=False, ignore_projections=False):
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
    set_c_is_vector - boolean
      Flag to indicate if set_c is a vector, as opposed to a gdal_dataset.
    ignore_projections - boolean
      A flag to ignore projection differences between feature and response sets - use only if you 
      are sure the projections are really the same.


    Return: 
    None, simply throw error if the check fails
    """
    if (len(set_a) != len(set_b)):
        raise Exception('different number of training features and responses')
    if (len(set_c) > 0):
        if (len(set_a) != len(set_c)):
            raise Exception('different number of training features and boundary files - give None for blank boundary')

    for n in range(0, len(set_a)):
        a_proj = get_proj(set_a[n], False)
        b_proj = get_proj(set_b[n], set_b_is_vector)

        if (a_proj != b_proj and ignore_projections == False):
            raise Exception(('projection mismatch between', set_a[n], 'and', set_b[n]))

        if (len(set_c) > 0):
            if (set_c[n] is not None):
                c_proj = get_proj(set_c[n], set_c_is_vector)
            else:
                c_proj = b_proj

            if (a_proj != c_proj and ignore_projections == False):
                raise Exception(('projection mismatch between', set_a[n], 'and', set_c[n]))

        if (set_b_is_vector == False):
            dataset_a = gdal.Open(set_a[n], gdal.GA_ReadOnly)
            dataset_b = gdal.Open(set_b[n], gdal.GA_ReadOnly)

            if (dataset_a.GetProjection() != dataset_b.GetProjection() and ignore_projections == False):
                raise Exception(('projection mismatch between', set_a[n], 'and', set_b[n]))

            if (dataset_a.GetGeoTransform() != dataset_b.GetGeoTransform()):
                raise Exception(('geotransform mismatch between', set_a[n], 'and', set_b[n]))

            if (dataset_a.RasterXSize != dataset_b.RasterXSize or dataset_a.RasterYSize != dataset_b.RasterYSize):
                raise Exception(('extent mismatch between', set_a[n], 'and', set_b[n]))

        if (len(set_c) > 0):
            if (set_c[n] is not None and set_c_is_vector == False):
                dataset_a = gdal.Open(set_a[n], gdal.GA_ReadOnly)
                dataset_c = gdal.Open(set_c[n], gdal.GA_ReadOnly)

                if (dataset_a.GetProjection() != dataset_c.GetProjection() and ignore_projections == False):
                    raise Exception(('projection mismatch between', set_a[n], 'and', set_c[n]))

                if (dataset_a.GetGeoTransform() != dataset_c.GetGeoTransform()):
                    raise Exception(('geotransform mismatch between', set_a[n], 'and', set_c[n]))

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

    if (config.data_save_name is not None):
        config.saved_data = False
        config.save_to_file()
        for _w in range(len(weights)):
            np.save(config.weight_files[_w], weights[_w])
        config.saved_data = True
        config.save_to_file()

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


#TODO: break into more usable pieces


def build_training_data_ordered(config):
    """ Builds a set of training data based on the configuration input
        Arguments:
        config - object of type Data_Config with the requisite values for preparing training data (see __init__.py)


        Return: 
        features - 4d numpy array 
          Array of data features, arranged as n,y,x,p, where n is the number of samples, y is the 
          data y dimension (2*window_radius), x is the data x dimension (2*window_radius), 
          and p is the number of features.
        responses - 4d numpy array
          Array of of data responses, arranged as n,y,x,p, where n is the number of samples, y is the 
          data y dimension (2*window_radius), x is the data x dimension (2*window_radius), 
          and p is the number of responses.  Each slice in the response dimension is a binary array o
          f that response class value.
        training_fold - numpy array 
          An array indicating which sample belongs to which data fold, from 0 to n_folds-1.
    """

    assert config.raw_feature_file_list is not [], 'feature files to pull data from are required'
    assert config.raw_response_file_list is not [], 'response files to pull data from are required'

    if (config.random_seed is not None):
        np.random.seed(config.random_seed)

    # TODO: relax reading style to not force all of these conditions (e.g., flex extents, though proj and px size should match for now)
    check_data_matches(config.raw_feature_file_list, config.raw_response_file_list, False,
                       config.boundary_file_list, config.boundary_as_vectors, config.ignore_projections)

    if (isinstance(config.max_samples, list)):
        if (len(config.max_samples) != len(config.raw_feature_file_list)):
            raise Exception('max_samples must equal feature_file_list length, or be an integer.')

    features = []
    responses = []
    repeat_index = []

    n_features = np.nan

    for _i in range(0, len(config.raw_feature_file_list)):
        # TODO: external update through loggers

        # open requisite datasets
        dataset = gdal.Open(config.raw_feature_file_list[_i], gdal.GA_ReadOnly)
        if (np.isnan(n_features)):
            n_features = dataset.RasterCount
        feature = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount))
        for n in range(0, dataset.RasterCount):
            feature[:, :, n] = dataset.GetRasterBand(n+1).ReadAsArray()

        response = gdal.Open(config.raw_response_file_list[_i]).ReadAsArray().astype(float)
        if (config.response_min_value is not None):
            response[response < config.response_min_value] = config.response_nodata_value
        if (config.response_max_value is not None):
            response[response > config.response_max_value] = config.response_nodata_value

        if (len(config.boundary_file_list) > 0):
            if (config.boundary_file_list[n] is not None):
                if (config.boundary_as_vectors):
                    mask = rasterize_vector(config.boundary_file_list[_i], dataset.GetGeoTransform(), [
                                            feature.shape[0], feature.shape[1]])
                else:
                    mask = gdal.Open(config.boundary_file_list[_i]).ReadAsArray().astype(float)
                feature[mask == config.boundary_bad_value, :] = config.feature_nodata_value
                response[mask == config.boundary_bad_value] = config.response_nodata_value

        # TODO: log feature shape
        # _logger.trace(whatever)

        # ensure nodata values are consistent
        if (not dataset.GetRasterBand(1).GetNoDataValue() is None):
            feature[feature == dataset.GetRasterBand(1).GetNoDataValue()] = config.feature_nodata_value
        feature[np.isnan(feature)] = config.feature_nodata_value
        feature[np.isinf(feature)] = config.feature_nodata_value

        # TODO: consider whether we want the lines below included (or if we want them as an option)
        response[feature[:, :, 0] == config.feature_nodata_value] = config.response_nodata_value
        feature[response == config.response_nodata_value, :] = config.feature_nodata_value

        pos_len = 0
        cr = [0, feature.shape[1]]
        rr = [0, feature.shape[0]]

        collist = [x for x in range(cr[0]+config.window_radius, cr[1] -
                                    config.window_radius, config.internal_window_radius*2)]
        rowlist = [x for x in range(rr[0]+config.window_radius, rr[1] -
                                    config.window_radius, config.internal_window_radius*2)]

        colrow = np.zeros((len(collist)*len(rowlist), 2)).astype(int)
        colrow[:, 0] = np.matlib.repmat(np.array(collist).reshape((-1, 1)), 1, len(rowlist)).flatten()
        colrow[:, 1] = np.matlib.repmat(np.array(rowlist).reshape((1, -1)), len(collist), 1).flatten()
        del collist
        del rowlist

        colrow = colrow[np.random.permutation(colrow.shape[0]), :]

        # TODO: replace tqdm with log
        for _cr in tqdm(range(len(colrow)), ncols=80):
            col = colrow[_cr, 0]
            row = colrow[_cr, 1]

            r = response[row-config.window_radius:row+config.window_radius,
                         col-config.window_radius:col+config.window_radius].copy()
            if(r.shape[0] == config.window_radius*2 and r.shape[1] == config.window_radius*2):
                if ((np.sum(r == config.response_nodata_value) <= r.size * config.nodata_maximum_fraction)):
                    d = feature[row-config.window_radius:row+config.window_radius,
                                col-config.window_radius:col+config.window_radius].copy()

                    responses.append(r)
                    features.append(d)
                    pos_len += 1

                    if (pos_len >= config.max_samples):
                        break

    # stack images up
    features = np.stack(features)
    responses = np.stack(responses)

    # randombly permute data to reshuffle everything
    perm = np.random.permutation(features.shape[0])
    features = features[perm, :]
    responses = responses[perm, :]
    fold_assignments = np.zeros(responses.shape[0]).astype(int)

    for f in range(0, config.n_folds):
        idx_start = int(round(f / config.n_folds * len(fold_assignments)))
        idx_finish = int(round((f + 1) / config.n_folds * len(fold_assignments)))
        fold_assignments[idx_start:idx_finish] = f

    # reshape images for the CNN
    features = features.reshape((features.shape[0], features.shape[1], features.shape[2], n_features))
    responses = responses.reshape((responses.shape[0], responses.shape[1], responses.shape[2], 1))

    weights = np.ones((responses.shape[0], responses.shape[1], responses.shape[2], 1))
    weights[responses[..., 0] == config.response_nodata_value] = 0

    if (config.internal_window_radius != config.window_radius):
        buf = config.window_radius - config.internal_window_radius
        weights[:, :buf, :, -1] = 0
        weights[:, -buf:, :, -1] = 0
        weights[:, :, :buf, -1] = 0
        weights[:, :, -buf:, -1] = 0

    _logger.debug('Feature shape: {}'.format(features.shape))
    _logger.debug('Response shape: {}'.format(response.shape))

    config.response_shape = responses.shape
    config.feature_shape = features.shape

    if (config.data_build_category == 'ordered_categorical'):
        # Lists are a hack so that the calculate_categorical_weights can be written for external
        # use cases
        un_resp = np.unique(responses)
        un_resp = un_resp[un_resp != config.response_nodata_value]
        cat_responses = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2], len(un_resp)))
        for _r in range(len(un_resp)):
            cat_responses[..., _r] = np.squeeze(responses == un_resp[_r])
        responses = cat_responses

    features = [features[fold_assignments == fold, ...] for fold in range(config.n_folds)]
    responses = [responses[fold_assignments == fold, ...] for fold in range(config.n_folds)]
    weights = [weights[fold_assignments == fold, ...] for fold in range(config.n_folds)]
    if (config.data_build_category == 'ordered_categorical'):
        weights = calculate_categorical_weights(responses, weights, config)

    if (config.data_save_name is not None):
        for fold in range(config.n_folds):
            np.save(config.feature_files[fold], features[fold])
            np.save(config.response_files[fold], responses[fold])
            np.save(config.weight_files[fold], weights[fold])
        #TODO: remove below and touch data_save_file
        config.saved_data = True
        config.save_to_file()

    return features, responses, weights
