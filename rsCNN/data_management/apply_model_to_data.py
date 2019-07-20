import os

import gdal
import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm

from rsCNN.data_management import common_io, ooc_functions
from rsCNN.data_management.data_core import DataContainer

plt.switch_backend('Agg')  # Needed for remote server plotting


def apply_model_to_raster(
        cnn: keras.Model,
        data_container: DataContainer,
        feature_file: str,
        destination_basename: str,
        make_png: bool = False,
        make_tif: bool = True,
        CNN_MODE: bool = False,
        exclude_feature_nodata: bool = False
) -> None:
    """ Apply a trained model to a raster file.

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

    config = data_container.config

    assert os.path.dirname(destination_basename), 'Output directory does not exist'

    # Open feature dataset and establish n_classes
    feature_set = gdal.Open(feature_file, gdal.GA_ReadOnly)
    n_classes = cnn.predict(
        np.zeros((1, config.data_build.window_radius*2, config.data_build.window_radius*2, feature_set.RasterCount))).shape[-1]

    # Initialize Output Dataset
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    outDataset = driver.Create(destination_basename + '.tif',
                               feature_set.RasterXSize,
                               feature_set.RasterYSize,
                               n_classes,
                               gdal.GDT_Float32)

    outDataset.SetProjection(feature_set.GetProjection())
    outDataset.SetGeoTransform(feature_set.GetGeoTransform())

    step_size = config.data_build.loss_window_radius*2
    if (CNN_MODE):
        step_size = 1
        internal_offset = config.data_build.loss_window_radius - 1
    else:
        internal_offset = config.data_build.window_radius - config.data_build.loss_window_radius

    # Find the UL indicies of all prediction locations
    cr = [0, feature_set.RasterXSize]
    rr = [0, feature_set.RasterYSize]

    collist = [x for x in range(cr[0], cr[1] - 2*config.data_build.window_radius, step_size)]
    rowlist = [x for x in range(rr[0], rr[1] - 2*config.data_build.window_radius, step_size)]
    collist.append(cr[1]-2*config.data_build.window_radius)
    rowlist.append(rr[1]-2*config.data_build.window_radius)

    for _c in tqdm(range(len(collist)), ncols=80, desc='Applying model to scene'):
        col = collist[_c]
        images = []

        write_ul = []
        for row in rowlist:
            d, m = common_io.read_map_subset([feature_file],
                                             [[col, row]],
                                             config.data_build.window_radius * 2,
                                             mask=None,
                                             nodata_value=config.raw_files.feature_nodata_value)
            if (d is None):
                continue

            if(d.shape[0] == config.data_build.window_radius*2 and d.shape[1] == config.data_build.window_radius*2):
                # TODO: consider having this as an option
                # d = fill_nearest_neighbor(d)
                images.append(d)
                write_ul.append([col + internal_offset, row + internal_offset])
        images = np.stack(images)
        images = images.reshape((images.shape[0], images.shape[1], images.shape[2], feature_set.RasterCount))

        images, image_band_types = ooc_functions.one_hot_encode_array(data_container.feature_raw_band_types, images)

        if (config.data_build.feature_mean_centering is True):
            images -= np.nanmean(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]

        if (data_container.feature_scaler is not None):
            images = data_container.feature_scaler.transform(images)

        images[np.isnan(images)] = config.data_samples.feature_nodata_encoding
        pred_y = cnn.predict(images)
        if (data_container.response_scaler is not None):
            pred_y = data_container.response_scaler.inverse_transform(pred_y)

        pred_y[np.logical_not(np.isfinite(pred_y))] = config.raw_files.response_nodata_value

        if (exclude_feature_nodata):
            nd_set = np.all(np.isnan(images), axis=-1)
            pred_y[nd_set, ...] = config.raw_files.response_nodata_value

        if (not CNN_MODE):
            if (internal_offset != 0):
                pred_y = pred_y[:, internal_offset:-internal_offset, internal_offset:-internal_offset, :]

        for _b in range(0, n_classes):
            for _i in range(len(images)):
                if CNN_MODE:
                    outDataset.GetRasterBand(
                        _b+1).WriteArray(pred_y[_i, _b].reshape((1, 1)), write_ul[_i][0], write_ul[_i][1])
                else:
                    outDataset.GetRasterBand(_b+1).WriteArray(pred_y[_i, :, :, _b], write_ul[_i][0], write_ul[_i][1])
        outDataset.FlushCache()

    # if (make_png):
    #    output[output == config.response_nodata_value] = np.nan
    #    feature[feature == config.response_nodata_value] = np.nan
    #    gs1 = gridspec.GridSpec(1, n_classes+1)
    #    for n in range(0, n_classes):
    #        ax = plt.subplot(gs1[0, n])
    #        im = plt.imshow(output[:, :, n], vmin=0, vmax=1)
    #        plt.axis('off')

    #    ax = plt.subplot(gs1[0, n_classes])
    #    im = plt.imshow(np.squeeze(feature[..., 0]))
    #    plt.axis('off')
    #    plt.savefig(destination_basename + '.png', dpi=png_dpi, bbox_inches='tight')
    #    plt.clf()


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
