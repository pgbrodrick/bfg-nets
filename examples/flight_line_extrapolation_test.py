import argparse
import os
import sys
import gdal
import numpy as np

# TODO manage imports
from rsCNN import networks
from rsCNN.data_management import DataConfig, training_data, transforms, apply_model_to_data


# TODO script needs to be adapted yet

parser = argparse.ArgumentParser(description='CNN example for spatial extrapolation from CAO flight lines')
parser.add_argument('key')
args = parser.parse_args()

window_radius = 16

year = '2015'
feature_files = ['../global_cwc/dat/features/feat_subset.tif']
response_files = ['../global_cwc/dat/responses/resp_subset.tif']

# TODO:, grab from a config somewhere
verification_fold = 0

# could (and typically are) different, but using for now
application_feature_files = feature_files
application_output_basenames = ['examples/output/' +
                                os.path.basename(x).split('.')[0] + '_applied_cnn' for x in feature_files]

# TODO: if we want to grab these from a config file, need to write a wrapper to read the transform in
data_options = {
    'data_save_name': 'examples/munged/cnn_munge_' + str(window_radius) + '_test',
    'max_samples': 30000,
    'internal_window_radius': int(round(window_radius*0.5)),
    'response_max_value': 10000,
    'response_min_value': 0
}

data_config = DataConfig(window_radius, feature_files, response_files, **data_options)


if (args.key == 'build' or args.key == 'all'):
    features, responses, fold_assignments = training_data.build_regression_training_data_ordered(data_config)
else:
    data_config = None  # TODO: load data_config from disk


network_options = {
    'use_batch_norm': False,
    'conv_depth': 16,
    'block_structure': (1,1),
    'output_activation': 'softplus',

    'network_name': 'cwc_test_network',
    'optimzer': 'adam',
    'batch_size': 100,
    'max_epochs': 10,
    'n_noimprovement_repeats': 30,
    'output_directory': None,
}

loss_function = networks.losses.cropped_loss('mae', features.shape[1], data_config.internal_window_radius*2)
network_config = networks.network_config.create_network_config('unet',
                                                               'test_spatial_model',
                                                               features.shape[1:],
                                                               1,
                                                               'mae',
                                                               **network_options)

cnn = networks.model.CNN(network_config)


# TODO add option for plotting training data previews

if (args.key == 'train' or args.key == 'all'):

    feature_scaler = transforms.RobustTransformer(
        data_config.feature_nodata_value, data_config.data_save_name + '_feature_')
    response_scaler = transforms.StandardTransformer(
        data_config.response_nodata_value, data_config.data_save_name + '_response_')

    print(np.mean(responses[responses[..., 0] != -9999, 0]))
    train_set = fold_assignments == verification_fold
    test_set = np.logical_not(train_set)
    feature_scaler.fit(features[train_set, ...])
    response_scaler.fit(responses[train_set, ..., :-1])

    features = feature_scaler.transform(features)
    responses[..., :-1] = response_scaler.transform(responses[..., :-1])
    print(np.mean(responses[responses[..., 0] != -9999, 0]))

    cnn.fit(features[train_set,...], 
            responses[train_set,...], 
            validation_data=(features[test_set,...],responses[test_set,...]))


if (args.key == 'apply' or args.key == 'all'):
    for _f in range(len(application_feature_files)):
        apply_model_to_data.apply_model_to_raster(cnn,
                                                  data_config,
                                                  application_feature_files[_f],
                                                  application_output_basenames[_f],
                                                  make_png=False,
                                                  make_tif=True,
                                                  feature_transformer=feature_scaler,
                                                  response_transformer=response_scaler)
