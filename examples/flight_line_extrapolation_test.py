import argparse
import os
import sys
import gdal
import numpy as np

# TODO manage imports
from rsCNN import networks
import rsCNN.data_management
import rsCNN.data_management.training_data
import rsCNN.data_management.transforms
import rsCNN.data_management.apply_model_to_data
import rsCNN.evaluation


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
config_options = {
    'data_save_name': 'examples/munged/cnn_munge_' + str(window_radius) + '_test',
    'max_samples': 30000,
    'internal_window_radius': int(round(window_radius*0.5)),
    'response_max_value': 10000,
    'response_min_value': 0,

    'use_batch_norm': True,
    'conv_depth': 'growth',
    'block_structure': (1, 1),
    'output_activation': 'softplus',

    'network_name': 'cwc_test_network2',
    'optimzer': 'adam',
    'batch_size': 100,
    'max_epochs': 100,
    'n_noimprovement_repeats': 10,
    'output_directory': None
}


if (args.key == 'data' or args.key == 'all'):
    data_config = rsCNN.data_management.DataConfig(window_radius, feature_files, response_files, **config_options)
    features, responses, fold_assignments = rsCNN.data_management.training_data.build_regression_training_data_ordered(
        data_config)
else:
    data_config = rsCNN.data_management.load_data_config_from_file(config_options['data_save_name'])


loss_function = networks.losses.cropped_loss(
    'mae', data_config.feature_shape[1], config_options['internal_window_radius']*2)
network_config = networks.network_config.create_network_config('unet',
                                                               config_options['network_name'],
                                                               data_config.feature_shape[1:],
                                                               data_config.response_shape[-1] - 1,
                                                               'mae',
                                                               **config_options)

cnn = networks.model.CNN(network_config)


feature_scaler = rsCNN.data_management.transforms.RobustTransformer(
    data_config.feature_nodata_value, data_config.data_save_name + '_feature_')
response_scaler = rsCNN.data_management.transforms.StandardTransformer(
    data_config.response_nodata_value, data_config.data_save_name + '_response_')

if (args.key == 'train' or 'model_eval'):
    npzf = np.load(data_config.data_save_name + '.npz')
    features = npzf['features']
    responses = npzf['responses']
    fold_assignments = npzf['fold_assignments']

if (args.key == 'train' or args.key == 'all'):


    train_set = fold_assignments == verification_fold
    test_set = np.logical_not(train_set)
    feature_scaler.fit(features[train_set, ...])
    response_scaler.fit(responses[train_set, ..., :-1])

    feature_scaler.save()
    response_scaler.save()

    features = feature_scaler.transform(features)
    responses[..., :-1] = response_scaler.transform(responses[..., :-1])

    cnn.fit(feature_scaler.transform(features[train_set, ...]),
            response_scaler.transform(responses[train_set, ...]),
            validation_data=(feature_scaler.transform(features[test_set, ...]), 
            response_scaler.transform(responses[test_set, ...])))


if (args.key == 'apply' or args.key == 'all'):
    feature_scaler.load()
    response_scaler.load()
    for _f in range(len(application_feature_files)):
        rsCNN.data_management.apply_model_to_data.apply_model_to_raster(cnn,
                              data_config,
                              application_feature_files[_f],
                              application_output_basenames[_f],
                              make_png=False,
                              make_tif=True,
                              feature_transformer=feature_scaler,
                              response_transformer=response_scaler)


if (args.key == 'model_eval' or args.key == 'all'):
    feature_scaler.load()
    response_scaler.load()

    rsCNN.evaluation.generate_eval_report(cnn,'examples/output/test_model_eval.pdf',
                                          features,
                                          responses[...,:-1],
                                          responses[...,-1],
                                          fold_assignments,
                                          verification_fold,
                                          feature_scaler,
                                          response_scaler,
                                          data_config)

