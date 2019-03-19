from rsCNN.utils.general import *
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import gdal


def apply_model_to_raster(cnn, data_config, feature_file, destination_basename, make_png=False, make_tif=True, feature_transformer=None, response_transformer=None):
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

    if (feature_transformer is not None):
        feature = feature_transformer.transform(feature)

    output = np.zeros((feature.shape[0], feature.shape[1], cnn.network_config['architecture']['n_classes'])) + data_config.response_nodata_value

    cr = [0, feature.shape[1]]
    rr = [0, feature.shape[0]]

    collist = [x for x in range(cr[0]+data_config.window_radius, cr[1] -
                                data_config.window_radius, data_config.internal_window_radius*2)]
    collist.append(cr[1]-data_config.window_radius)
    rowlist = [x for x in range(rr[0]+data_config.window_radius, rr[1] -
                                data_config.window_radius, data_config.internal_window_radius*2)]
    rowlist.append(rr[1]-data_config.window_radius)

    for col in collist:
        images = []
        for n in rowlist:
            d = feature[n-data_config.window_radius:n+data_config.window_radius,
                        col-data_config.window_radius:col+data_config.window_radius].copy()
            if(d.shape[0] == data_config.window_radius*2 and d.shape[1] == data_config.window_radius*2):
                # TODO: implement local scaling if necessary

                # TODO: consider having this as an option
                # d = fill_nearest_neighbor(d)
                images.append(d)
        images = np.stack(images)
        images = images.reshape((images.shape[0], images.shape[1], images.shape[2], dataset.RasterCount))

        pred_y = cnn.predict(images)

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
    output[np.all(feature == data_config.feature_nodata_value, axis=-1), :] = data_config.response_nodata_value
    if (feature_transformer is not None):
        output = response_transformer.inverse_transform(output)

    if (make_png):
        output[output == data_config.response_nodata_value] = np.nan
        feature[feature == data_config.response_nodata_value] = np.nan
        gs1 = gridspec.GridSpec(1, cnn.network_config['architecture']['n_classes']+1)
        for n in range(0, cnn.network_config['architecture']['n_classes']):
            ax = plt.subplot(gs1[0, n])
            im = plt.imshow(output[:, :, n], vmin=0, vmax=1)
            plt.axis('off')

        ax = plt.subplot(gs1[0, cnn.network_config['architecture']['n_classes']])
        im = plt.imshow(np.squeeze(feature[..., 0]))
        plt.axis('off')
        plt.savefig(destination_basename + '.png', dpi=png_dpi, bbox_inches='tight')
        plt.clf()

    if (make_tif):
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        output[np.isnan(output)] = data_config.response_nodata_value

        outDataset = driver.Create(destination_basename + '.tif',
                                   output.shape[1], output.shape[0], cnn.network_config['architecture']['n_classes'], gdal.GDT_Float32)
        outDataset.SetProjection(dataset.GetProjection())
        outDataset.SetGeoTransform(dataset.GetGeoTransform())
        for n in range(0, cnn.network_config['architecture']['n_classes']):
            outDataset.GetRasterBand(n+1).WriteArray(np.squeeze(output[:, :, n]), 0, 0)
        del outDataset
    del dataset
