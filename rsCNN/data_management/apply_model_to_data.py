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

from rsCNN.data_management import common_io, ooc_functions
from rsCNN.data_management.data_core import DataContainer

plt.switch_backend('Agg')  # Needed for remote server plotting

_logger = logging.getLogger(__name__)

def apply_model_to_raster(
        cnn: keras.Model,
        data_container: DataContainer,
        feature_files: List[str],
        destination_basename: str,
        output_format: str = 'GTiff',
        creation_options: List[str] = None,
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
    if creation_options is None:
        creation_options = []

    config = data_container.config

    assert os.path.dirname(destination_basename), 'Output directory does not exist'

    valid_output_formats = ['GTiff','ENVI']
    err_str = 'Not a viable output format, options are: {}'.format(valid_output_formats)
    assert output_format in valid_output_formats, err_str

    # Open feature dataset and establish n_classes
    feature_sets = [[gdal.Open(f, gdal.GA_ReadOnly) for f in feature_files]]
    internal_ul_list, x_len, y_len = common_io.get_overlapping_extent(feature_sets)

    assert x_len > 0 and y_len > 0, 'No common feature file interior'

    n_classes = len(data_container.response_band_types)

    # Initialize Output Dataset
    driver = gdal.GetDriverByName(output_format)
    driver.Register()

    temporary_outname = destination_basename
    if (output_format != 'ENVI'):
        temporary_outname = temporary_outname + '_temporary_ENVI_file'
    outDataset = driver.Create(temporary_outname, x_len, y_len, n_classes, gdal.GDT_Float32)
    out_trans = list(feature_sets[0][0].GetGeoTransform())
    out_trans[0] = out_trans[0] + internal_ul_list[0][0]*out_trans[1]
    out_trans[3] = out_trans[3] + internal_ul_list[0][0]*out_trans[5]
    outDataset.SetProjection(feature_sets[0][0].GetProjection())
    outDataset.SetGeoTransform(out_trans)
    del outDataset

    for _row in range(0,y_len, config.data_build.loss_window_radius*2):
        col_dat = _read_chunk_by_row(feature_sets, internal_ul_list, x_len, config.data_build.window_radius*2, _row,
                             data_container.feature_raw_band_types)

        tile_dat, x_index = _convert_chunk_to_tiles(col_dat, config.loss_window_radius, config.window_radius)
        tile_dat = ooc_functions.one_hot_encode_array(data_container.feature_raw_band_types, tile_dat,
                                                      data_container.feature_per_band_encoded_values)

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
            pred_y[nd_set, ...] = config.raw_files.response_nodata_value
        del tile_dat

        window_radius_difference = config.loss_window_radius - config.window_radius
        if (window_radius_difference > 0):
            pred_y = pred_y[:,window_radius_difference:-window_radius_difference,window_radius_difference:-window_radius_difference,:]
        output = np.zeros((config.data_build.loss_window_radius*2, x_len, pred_y.shape[-1]))
        for _c in x_index:
            output[:,_c:_c+config.loss_window_radius*2,:] = pred_y[_c,...]

        output_memmap = np.memmap(temporary_outname, mode='r+', shape=(y_len,x_len,n_classes), dtype=np.float32)
        outrow = _row + window_radius_difference
        output_memmap[outrow:outrow+config.loss_window_radius,:,:] = output
        del output_memmap

    if (output_format != 'ENVI'):
        final_outname = destination_basename
        if (output_format == 'GTiff'):
            final_outname += '.tif'

        cmd_str = 'gdal_translate {} {} -of {}'.format(temporary_outname, final_outname, output_format)
        for co in creation_options:
            cmd_str += ' -co {}'.format(co)
        subprocess.call(cmd_str, shell=True)
        test_outds = gdal.Open(final_outname,gdal.GA_ReadOnly)
        if (test_outds is not None):
            os.remove(temporary_outname)
            os.remove(temporary_outname + '.hdr')
            try:
                os.remove(temporary_outname + '.aux.xml')
            except OSError:
                pass
        else:
            _logger.error('Failed to successfully convert output ENVI file to {}.  ENVI file available at: {}'.
                          format(output_format, temporary_outname))


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
    for _col in range(0,feature_data.shape[1],loss_window_radius*2):
        if (feature_data.shape[1] - _col >= window_radius*2):
            col_index.append(_col)
        else:
            col_index.append(feature_data.shape[1]-window_radius*2)
        output_array.append(feature_data[:,col_index[-1]:col_index[-1]+window_radius*2,:])
    output_array = np.stack(output_array)
    output_array = np.reshape((output_array.shape[0],output_array.shape[1],output_array.shape[2],feature_data.shape[-1]))

    col_index = np.array(col_index)

    return output_array, col_index




def _read_chunk_by_row(feature_sets: List[gdal.dataset], pixel_upper_lefts: List[List[int]], x_size: int, y_size: int,
              line_offset: int, raw_band_types: List[str]) -> np.array:
    """
    Read a chunk of feature data line-by-line.

    Args:
        feature_sets: each feature dataset to read
        pixel_upper_lefts: upper left hand pixel of each dataset
        x_size: size of x data to read
        y_size: size of y data to read
        line_offset: line offset from UL of each set to start reading at
        raw_band_types: band types before encoding
        categorical_encodings: encodings of each categorical band type

    Returns:
        feature_array: feature array
    """

    total_features = len(raw_band_types)
    output_array = np.zeros((y_size, x_size,total_features))

    feat_ind = 0
    for _feat in range(len(feature_sets)):
        dat = np.zeros((feature_sets[_feat].RasterCount,y_size,x_size))
        for _line in range(line_offset,y_size):
            dat[:,_line:_line+1,:] = feature_sets[_feat].ReadAsArray(pixel_upper_lefts[_feat][0],
                                     pixel_upper_lefts[_feat][1] + _line,x_size,1).astype(np.float32)
        dat = np.swapaxes(dat,0,1)
        dat = np.swapaxes(dat,1,2)

        output_array[...,feat_ind:feat_ind+dat.shape[-1]] = dat
        feat_ind += dat.shape[-1]

    return output_array


def maximum_likelihood_classification(
        likelihood_file: str,
        output_file_base: str,
        make_png: bool = True,
        make_tif: bool = False,
        png_dpi: int = 200,
        output_nodata_value: int = -1
) -> None:
    """ Convert a n-band map of probabilities to a classified image using maximum likelihood.

    Args:
        cnn:  Pre-trained keras CNN model
        data_container: Holds info like scalers
        feature_file: File with feature data to apply the model to
        destination_basename: Base of the output file (will get a .tif or .png extention)

        make_png: Should an output be created in PNG format?
        make_tif: Should an output be created in TIF format?
        CNN_MODE: Should the model be applied in CNN mode?
        exclude_feature_nodata: Flag to remove all pixels in features without data from applied model

    :Returns
        None
    """

    output_tif_file = output_file_base + '.tif'
    output_png_file = output_file_base + '.png'

    dataset = gdal.Open(likelihood_file, gdal.GA_ReadOnly)

    output = np.zeros((dataset.RasterYSize, dataset.RasterXSize))
    output[dataset.GetRasterBand(1).ReadAsArray() == dataset.GetRasterBand(1).GetNoDataValue()] = output_nodata_value

    for line in tqdm(np.arange(0, dataset.RasterYSize).astype(int), ncols=80, desc='Calculating max likelihood'):
        prob = dataset.ReadAsArray(0, line, dataset.RasterXSize, 1)
        output[line, :] = np.argmax(prob)
        output[np.any(prob == output_nodata_value, axis=0)] = output_nodata_value

    if (make_tif):
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()

        outDataset = driver.Create(output_tif_file,
                                   output.shape[1],
                                   output.shape[0],
                                   1,
                                   gdal.GDT_Float32,
                                   options=['COMPRESS=LZW'])

        outDataset.SetProjection(dataset.GetProjection())
        outDataset.SetGeoTransform(dataset.GetGeoTransform())
        outDataset.GetRasterBand(1).WriteArray(output, 0, 0)
        del outDataset
    if (make_png):

        output[output == output_nodata_value] = np.nan
        cmap = mpl.cm.Set1_r
        cmap.set_bad('black', 1.)
        plt.imshow(output, cmap=cmap)
        plt.axis('off')
        plt.savefig(output_png_file, dpi=png_dpi, bbox_inches='tight')
        plt.clf()


