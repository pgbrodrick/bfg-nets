import logging
import os
from typing import List, Tuple

import gdal
import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import subprocess
from tqdm import tqdm

from bfgn.data_management import common_io, ooc_functions
from bfgn.data_management.data_core import DataContainer

plt.switch_backend('Agg')  # Needed for remote server plotting

_logger = logging.getLogger(__name__)


def apply_model_to_raster(
        cnn: keras.Model,
        data_container: DataContainer,
        feature_files: List[str],
        destination_basename: str,
        output_format: str = 'GTiff',
        creation_options: List[str] = [],
        CNN_MODE: bool = False,
        exclude_feature_nodata: bool = False
) -> None:
    """ Apply a trained model to a raster file.

    Args:
        cnn:  Pre-trained keras CNN model
        data_container: Holds info like scalers
        feature_files: Per-site feature files to apply the model to
        destination_basename: Base of the output file (will get appropriate extension)

        output_format: A viable gdal output data format.
        creation_options: GDAL creation options to pass for output file, e.g.: ['TILED=YES', 'COMPRESS=DEFLATE']
        CNN_MODE: Should the model be applied in CNN mode?
        exclude_feature_nodata: Flag to remove all pixels in features without data from applied model

    :Returns
        None
    """

    assert CNN_MODE is False, 'CNN mode application not yet supported'

    config = data_container.config

    assert os.path.dirname(destination_basename), 'Output directory does not exist'

    assert type(feature_files) is list, 'Feature files for given site must be provided as a list'

    valid_output_formats = ['GTiff', 'ENVI']
    err_str = 'Not a viable output format, options are: {}'.format(valid_output_formats)
    assert output_format in valid_output_formats, err_str

    _logger.debug('Open feature datasets')
    feature_sets = [gdal.Open(f, gdal.GA_ReadOnly) for f in feature_files]

    _logger.debug('Check file validity')
    invalid_files = ''
    for _f in range(len(feature_sets)):
        if feature_sets[_f] is None:
            invalid_files += 'Invalid file: {}\n'.format(feature_files[_f])
    assert invalid_files == '', invalid_files

    _logger.debug('Get common feature file interior')
    [internal_ul_list], x_len, y_len = common_io.get_overlapping_extent([feature_sets])

    assert x_len > 0 and y_len > 0, 'No common feature file interior'

    n_classes = len(data_container.response_band_types)

    # Initialize Output Dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    temporary_outname = destination_basename
    if (output_format != 'ENVI'):
        temporary_outname = temporary_outname + '_temporary_ENVI_file'
        _logger.debug('Creating output envi file: {}'.format(temporary_outname))
    else:
        _logger.debug('Creating temporary output envi file: {}'.format(temporary_outname))
    outDataset = driver.Create(temporary_outname, x_len, y_len, n_classes, gdal.GDT_Float32)
    out_trans = list(feature_sets[0].GetGeoTransform())
    out_trans[0] = out_trans[0] + internal_ul_list[0][0]*out_trans[1]
    out_trans[3] = out_trans[3] + internal_ul_list[0][0]*out_trans[5]
    outDataset.SetProjection(feature_sets[0].GetProjection())
    outDataset.SetGeoTransform(out_trans)
    del outDataset

    for _row in tqdm(range(0, y_len, config.data_build.loss_window_radius*2), ncols=80):
        _row = min(_row, y_len - 2*config.data_build.window_radius)
        col_dat = _read_chunk_by_row(feature_sets, internal_ul_list, x_len, config.data_build.window_radius*2, _row)
        _logger.debug('Read data chunk of shape: {}'.format(col_dat.shape))

        tile_dat, x_index = _convert_chunk_to_tiles(
            col_dat, config.data_build.loss_window_radius, config.data_build.window_radius)
        _logger.debug('Data tiled to shape: {}'.format(tile_dat.shape))

        tile_dat = ooc_functions.one_hot_encode_array(data_container.feature_raw_band_types, tile_dat,
                                                      per_band_encoding=data_container.feature_per_band_encoded_values)
        _logger.debug('Data one_hot_encoded.  New shape: {}'.format(tile_dat.shape))

        if (config.data_build.feature_mean_centering is True):
            tile_dat -= np.nanmean(tile_dat, axis=(1, 2))[:, np.newaxis, np.newaxis, :]

        if (data_container.feature_scaler is not None):
            tile_dat = data_container.feature_scaler.transform(tile_dat)

        tile_dat[np.isnan(tile_dat)] = config.data_samples.feature_nodata_encoding

        pred_y = cnn.predict(tile_dat)
        if (data_container.response_scaler is not None):
            pred_y = data_container.response_scaler.inverse_transform(pred_y)

        if (exclude_feature_nodata):
            nd_set = np.all(np.isnan(tile_dat), axis=-1)
            pred_y[nd_set, ...] = config.data_build.raw_files.response_nodata_value
        del tile_dat

        window_radius_difference = config.data_build.window_radius - config.data_build.loss_window_radius
        _logger.debug('Creating output dataset, using window_radius_difference {}'.format(window_radius_difference))
        if (window_radius_difference > 0):
            pred_y = pred_y[:, window_radius_difference:-window_radius_difference,
                            window_radius_difference:-window_radius_difference, :]
        output = np.zeros((config.data_build.loss_window_radius*2, x_len, pred_y.shape[-1]))
        for _c in range(len(x_index)):
            output[:, x_index[_c]+window_radius_difference:x_index[_c]+window_radius_difference +
                   config.data_build.loss_window_radius*2, :] = pred_y[_c, ...]

        _logger.debug('Convert output shape from (y,x,b) to (b,y,x)')
        output = np.moveaxis(output, [0, 1, 2], [1, 2, 0])

        output_memmap = np.memmap(temporary_outname, mode='r+', shape=(n_classes, y_len, x_len), dtype=np.float32)
        outrow = _row + window_radius_difference
        output_memmap[:, outrow:outrow+config.data_build.loss_window_radius*2, :] = output
        del output_memmap

    common_io.convert_envi_file(temporary_outname, destination_basename, output_format, True, creation_options)


def maximum_likelihood_classification(
        likelihood_file: str,
        data_container: DataContainer,
        destination_basename: str,
        output_format: str = 'GTiff',
        creation_options: List[str] = [],
) -> None:
    """ Convert a n-band map of probabilities to a classified image using maximum likelihood.

    Args:
        likelihood_file: File with per-class likelihoods
        data_container: Holds info like scalers
        destination_basename: Base of the output file (will get appropriate extension)
        creation_options: GDAL creation options to pass for output file, e.g.: ['TILED=YES', 'COMPRESS=DEFLATE']

    :Returns
        None
    """


    dataset = gdal.Open(likelihood_file, gdal.GA_ReadOnly)
    assert dataset is not None, 'Invalid liklihood_file'

    # Initialize Output Dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    temporary_outname = destination_basename
    if (output_format != 'ENVI'):
        temporary_outname = temporary_outname + '_temporary_ENVI_file'
        _logger.debug('Creating output envi file: {}'.format(temporary_outname))
    else:
        _logger.debug('Creating temporary output envi file: {}'.format(temporary_outname))

    outDataset = driver.Create(temporary_outname, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Int16)
    outDataset.SetProjection(dataset.GetProjection())
    outDataset.SetGeoTransform(dataset.GetGeoTransform())
    del outDataset

    for _row in tqdm(range(0, dataset.RasterYSize), ncols=80):
        col_dat = _read_chunk_by_row([dataset], [[0,0]], dataset.RasterXSize, 1, _row)
        _logger.debug('Read data chunk of shape: {}'.format(col_dat.shape))

        col_dat = np.argmax(col_dat, axis=-1)
        out_dat = np.zeros(col_dat.shape) + data_container.config.raw_files.response_nodata_value
        for _encoded_value in data_container.response_per_band_encoded_values[0]:
            out_dat[col_dat == _encoded_value] = data_container.response_per_band_encoded_values[0][_encoded_value]
        out_dat = np.reshape(out_dat,(out_dat.shape[0],out_dat.shape[1],1))

        _logger.debug('Convert output shape from (y,x,b) to (b,y,x)')
        out_dat = np.moveaxis(out_dat, [0, 1, 2], [1, 2, 0])

        output_memmap = np.memmap(temporary_outname, mode='r+', shape=(1, dataset.RasterYSize, dataset.RasterXSize),dtype=np.int16)
        output_memmap[:, _row:_row+1, :] = out_dat
        del output_memmap

    common_io.convert_envi_file(temporary_outname, destination_basename, output_format, True, creation_options)


def _convert_chunk_to_tiles(feature_data: np.array, loss_window_radius: int, window_radius: int) -> \
        Tuple[np.array, np.array]:
    """ Convert a set of rows to tiles to run through keras.  Assumes only one vertical layer is possible
    Args:
        feature_data:
        loss_window_radius:
        window_radius:
    Returns:
        output_array: array read to be passed through keras
        col_index: array of the left-coordinate of each output_array tile
    """

    output_array = []
    col_index = []
    for _col in range(0, feature_data.shape[1], loss_window_radius*2):
        col_index.append(min(_col, feature_data.shape[1]-window_radius*2))
        output_array.append(feature_data[:, col_index[-1]:col_index[-1]+window_radius*2, :])
    output_array = np.stack(output_array)
    output_array = np.reshape(
        output_array, (output_array.shape[0], output_array.shape[1], output_array.shape[2], feature_data.shape[-1]))

    col_index = np.array(col_index)

    return output_array, col_index


def _read_chunk_by_row(feature_sets: List[gdal.Dataset], pixel_upper_lefts: List[List[int]], x_size: int, y_size: int,
                       line_offset: int) -> np.array:
    """
    Read a chunk of feature data line-by-line.

    Args:
        feature_sets: each feature dataset to read
        pixel_upper_lefts: upper left hand pixel of each dataset
        x_size: size of x data to read
        y_size: size of y data to read
        line_offset: line offset from UL of each set to start reading at

    Returns:
        feature_array: feature array
    """

    total_features = np.sum([f.RasterCount for f in feature_sets])
    output_array = np.zeros((y_size, x_size, total_features))

    feat_ind = 0
    for _feat in range(len(feature_sets)):
        dat = np.zeros((feature_sets[_feat].RasterCount, y_size, x_size))
        for _line in range(y_size):
            dat[:, _line:_line+1, :] = feature_sets[_feat].ReadAsArray(int(pixel_upper_lefts[_feat][0]),
                                                                       int(pixel_upper_lefts[_feat][1] + line_offset + _line), x_size, 1).astype(np.float32)
        # Swap dat from (b,y,x) to (y,x,b)
        dat = np.moveaxis(dat, [0, 1, 2], [2, 0, 1])

        output_array[..., feat_ind:feat_ind+dat.shape[-1]] = dat
        feat_ind += dat.shape[-1]

    return output_array




