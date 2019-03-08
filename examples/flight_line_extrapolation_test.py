import apply_model
import train_model
import plot_utility
from src.util.general import *
from src.networks import CNN
from src.data_management import Data_Config, training_data, transforms
import rasterio.features
import ogr
import numpy as np
import matplotlib.pyplot as plt
import sys

import gdal
import matplotlib as mpl
mpl.use('Agg')

# TODO manage imports


# TODO script needs to be adapted yet

key = sys.argv[1]

window_radius = 16

year = '2015'
feature_files = ['dat/features/feat_subset.tif']
response_files = ['dat/responses/resp_subset.tif']
data_options = {
    'max_samples': 30000,
    'data_save_name': 'munged/cnn_munge_' + str(window_radius) + '_test',
    'internal_window_radius': rint(window_radius*0.5),
    'min_value': 0,
    'max_value': 10000,
}

data_config = Data_Config(window_radius, feature_files, response_files, **data_options)


network_options = {
    'conv_depth' = 16
    'batch_norm' = False
    'n_layers' = 10
    'conv_pattern' = 3
    'output_activation' = 'softplus'
    'network_name' = 'cwc_test_network'
}

training_options = {
    'batch_size' = 100
    'verification_fold' = 0
    'n_noimprovement_repeats' = 30
}



if (key == 'build' or key == 'all'):
    features, responses, fold_assignments = training_data.build_regression_training_data_ordered(data_config)

    feature_scaler = transforms.RobustScaler(nodata_value,config.savename)
    feature_scaler.fit(features)
    response_scaler = transforms.StandardScaler(nodata_value,config.savename)
    response_scaler.fit(responses)


# TODO add option for plotting training data previews

if (key == 'train' or key == 'all'):

    cnn = CNN()
    cnn.create_config('flat_regress_net', 
                      config.feature_shape[1:], 
                      n_classes=1, 
                      network_dictionary=network_config)

    cnn.fit(features,\
            responses,\
            fold_assignments,\
            **training_options)


if (key == 'apply' or key == 'all'):
    model = train_model.load_trained_model(model_name, window_radius, verbose=False, weighted=True, loss_type='mae')
    global_scale_file = munge_file + '_global_feature_scaling.npz'
    if (global_scaling == None):
        global_scale_file = None
    apply_model.apply_semantic_segmentation(feature_files,
                                            'output_maps',
                                            model,
                                            window_radius,
                                            internal_window_radius=internal_window_radius,
                                            local_scale_flag=local_scaling,
                                            global_scale_file=global_scale_file,
                                            make_png=False,
                                            make_tif=True,
                                            verbose=False,
                                            nodata_value=-9999)
