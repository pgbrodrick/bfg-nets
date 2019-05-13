import os

import gdal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from rsCNN import configs
from rsCNN.utils.general import *


plt.switch_backend('Agg')  # Needed for remote server plotting


def read_feature_chunk(feature_set, ul, window_size, nodata_value):

    subset = np.zeros((window_size, window_size, feature_set.RasterCount))
    for _b in range(feature_set.RasterCount):
        subset[..., _b] = feature_set.GetRasterBand(
            _b+1).ReadAsArray(int(ul[0]), int(ul[1]), int(window_size), int(window_size))

    subset[subset == nodata_value] = np.nan
    subset[np.logical_not(np.isfinite(subset))] = np.nan

    return subset


def apply_model_to_raster(cnn, config: configs.Config, feature_file, destination_basename, make_png=False, make_tif=True, feature_transformer=None, response_transformer=None, CNN_MODE=False):
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
        internal_offset = config.data_window_radius - config.data_build.loss_window_radius

    # Find the UL indicies of all prediction locations
    cr = [0, feature_set.RasterXSize]
    rr = [0, feature_set.RasterYSize]

    collist = [x for x in range(cr[0], cr[1] - 2*config.data_build.window_radius, step_size)]
    rowlist = [x for x in range(rr[0], rr[1] - 2*config.data_build.window_radius, step_size)]
    collist.append(cr[1]-2*config.data_build.window_radius)
    rowlist.append(rr[1]-2*config.data_build.window_radius)


    for _c in tqdm(range(len(collist)), ncols=80):
        col = collist[_c]
        images = []

        write_ul = []
        for row in rowlist:
            d = read_feature_chunk(feature_set, [col, row],
                                   config.data_build.window_radius*2, config.raw_files.feature_nodata_value)

            if(d.shape[0] == config.data_build.window_radius*2 and d.shape[1] == config.data_build.window_radius*2):
                # TODO: consider having this as an option
                # d = fill_nearest_neighbor(d)
                images.append(d)
                write_ul.append([col + internal_offset, row + internal_offset])
        images = np.stack(images)
        images = images.reshape((images.shape[0], images.shape[1], images.shape[2], feature_set.RasterCount))

        if (config.data_build.feature_mean_centering is True):
            images -= np.nanmean(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]

        if (feature_transformer is not None):
            images = feature_transformer.transform(images)

        images[np.isnan(images)] = config.data_samples.feature_nodata_encoding
        pred_y = cnn.predict(images)
        if (response_transformer is not None):
            pred_y = response_transformer.inverse_transform(pred_y)

        pred_y[np.logical_not(np.isfinite(pred_y))] = config.raw_files.response_nodata_value

        #nd_set = np.all(np.isnan(images), axis=-1)
        #pred_y[nd_set, ...] = data_config.response_nodata_value

        for _b in range(0, n_classes):
            for _i in range(len(images)):
                if CNN_MODE:
                    outDataset.GetRasterBand(
                        _b+1).WriteArray(pred_y[_i, _b].reshape((1, 1)), write_ul[_i][0], write_ul[_i][1])
                else:
                    outDataset.GetRasterBand(_b+1).WriteArray(pred_y[_i, :, :, _b], write_ul[_i][0], write_ul[_i][1])
        outDataset.FlushCache()

    # if (make_png):
    #    output[output == data_config.response_nodata_value] = np.nan
    #    feature[feature == data_config.response_nodata_value] = np.nan
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
