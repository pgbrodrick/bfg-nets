import os

import gdal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from rsCNN.utils.general import *


plt.switch_backend('Agg')  # Needed for remote server plotting


def apply_model_to_raster(cnn, data_config, feature_file, destination_basename, make_png=False, make_tif=True, feature_transformer=None, response_transformer=None, CNN_MODE=False):
    """ Apply a trained model to a raster file.

      Arguments:
      cnn - CNN (rsCNN/networks/__init__.py)
        Pre-trained model.
      data_config - DataConfig
        Tells us how the data was prepared for the cnn.
      feature_file - str
        File with feature data to apply the model to.
      destination_basename -  str
        Base of the output file (will get a .tif or .png extention).

      Keyword Arguments:
      make_png - boolean
        Should an output be created in PNG format?
      make_tif - boolean
        Should an output be created in TIF format?
      feature_transformer - object that inherets from BaseGlobalTransform
        If not none, this transform is applied to feature data before model application.
      response_transformer - object that inherets from BaseGlobalTransform
        If not none, this transform is applied to model output response data before writing tif/png.
    """

    assert os.path.dirname(destination_basename), 'Output directory does not exist'

    dataset = gdal.Open(feature_file, gdal.GA_ReadOnly)

    # TODO: implement an out-of core version here (IE, apply and write on the fly)
    feature = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount))
    for n in range(0, dataset.RasterCount):
        feature[:, :, n] = dataset.GetRasterBand(n+1).ReadAsArray()

    if (not dataset.GetRasterBand(1).GetNoDataValue() is None):
        feature[feature == dataset.GetRasterBand(1).GetNoDataValue()] = data_config.feature_nodata_value
    feature[np.isnan(feature)] = data_config.feature_nodata_value
    feature[np.isinf(feature)] = data_config.feature_nodata_value

    feature[feature == data_config.feature_nodata_value] = np.nan

    n_classes = cnn.predict(
        (np.zeros((1, data_config.window_radius*2, data_config.window_radius*2, feature.shape[-1])))).shape[-1]

    output = np.zeros((feature.shape[0], feature.shape[1], n_classes)) + data_config.response_nodata_value

    cr = [0, feature.shape[1]]
    rr = [0, feature.shape[0]]

    if (CNN_MODE):
        collist = [x for x in range(cr[0]+data_config.window_radius, cr[1] - data_config.window_radius)]
        rowlist = [x for x in range(rr[0]+data_config.window_radius, rr[1] - data_config.window_radius)]
    else:
        collist = [x for x in range(cr[0]+data_config.window_radius, cr[1] -
                                    data_config.window_radius, data_config.internal_window_radius*2)]
        rowlist = [x for x in range(rr[0]+data_config.window_radius, rr[1] -
                                    data_config.window_radius, data_config.internal_window_radius*2)]
    collist.append(cr[1]-data_config.window_radius)
    rowlist.append(rr[1]-data_config.window_radius)

    for _c in tqdm(range(len(collist)), ncols=80):
        col = collist[_c]
        images = []
        for n in rowlist:
            d = feature[n-data_config.window_radius:n+data_config.window_radius,
                        col-data_config.window_radius:col+data_config.window_radius].copy()
            if(d.shape[0] == data_config.window_radius*2 and d.shape[1] == data_config.window_radius*2):
                # TODO: consider having this as an option
                # d = fill_nearest_neighbor(d)
                images.append(d)
        images = np.stack(images)
        images = images.reshape((images.shape[0], images.shape[1], images.shape[2], dataset.RasterCount))

        if (data_config.feature_mean_centering is True):
            images -= np.nanmean(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]

        if (feature_transformer is not None):
            images = feature_transformer.transform(images)

        images[np.isnan(images)] = data_config.feature_training_nodata_value

        pred_y = cnn.predict(images)
        nd_set = np.all(np.isnan(images), axis=(1, 2, 3))
        pred_y[nd_set, ...] = data_config.response_nodata_value

        _i = 0
        for n in rowlist:
            p = pred_y[_i, ...]
            if (data_config.internal_window_radius < data_config.window_radius):
                mm = int(round(data_config.window_radius - data_config.internal_window_radius))
                p = p[mm:-mm, mm:-mm, :]
            output[n-data_config.internal_window_radius:n+data_config.internal_window_radius, col -
                   data_config.internal_window_radius:col+data_config.internal_window_radius, :] = p
            _i += 1
            if (_i >= len(images)):
                break

    # TODO: think through if this order of operations screws things up if response_nodata_value != feature_nodata_value
    if (feature_transformer is not None):
        output = response_transformer.inverse_transform(output)

    output[np.all(np.isnan(feature), axis=-1), :] = data_config.response_nodata_value

    if (make_png):
        output[output == data_config.response_nodata_value] = np.nan
        feature[feature == data_config.response_nodata_value] = np.nan
        gs1 = gridspec.GridSpec(1, n_classes+1)
        for n in range(0, n_classes):
            ax = plt.subplot(gs1[0, n])
            im = plt.imshow(output[:, :, n], vmin=0, vmax=1)
            plt.axis('off')

        ax = plt.subplot(gs1[0, n_classes])
        im = plt.imshow(np.squeeze(feature[..., 0]))
        plt.axis('off')
        plt.savefig(destination_basename + '.png', dpi=png_dpi, bbox_inches='tight')
        plt.clf()

    if (make_tif):
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        output[np.isnan(output)] = data_config.response_nodata_value

        outDataset = driver.Create(destination_basename + '.tif',
                                   output.shape[1], output.shape[0], n_classes, gdal.GDT_Float32)
        outDataset.SetProjection(dataset.GetProjection())
        outDataset.SetGeoTransform(dataset.GetGeoTransform())
        for n in range(0, n_classes):
            outDataset.GetRasterBand(n+1).WriteArray(np.squeeze(output[:, :, n]), 0, 0)
        del outDataset
    del dataset
