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


global_options = {
    'raw_feature_file_list' : ['../global_cwc/dat/features/feat_subset.tif'],
    'raw_response_file_list' : ['../global_cwc/dat/responses/resp_subset.tif'],

    'data_save_name': 'examples/munged/cwc_data_test',
    'max_samples': 30000,
    'window_radius' : 16,
    'internal_window_radius': 8,
    'response_max_value': 10000,
    'response_min_value': 0,

    'network_type': 'flat_regress_net',
    'use_batch_norm': True,
    'conv_depth': 'growth',
    'block_structure': (1,1),
    'n_layers': 10,
    'output_activation': 'softplus',
    'n_classes': 1,

    'network_name': 'cwc_test_network',
    'optimizer': 'adam',
    'batch_size': 100,
    'max_epochs': 100,
    'n_noimprovement_repeats': 10,
    'output_directory': None,
    'verification_fold': 0,
    'loss_metric': 'mae',

    'application_feature_files' : ['../global_cwc/dat/features/feat_subset.tif'],
    'application_output_basenames' : ['examples/output/feat_subset_applied_cnn.tif']
}



data_config = DataConfig( **global_options)

if (args.key == 'build' or args.key == 'all'):
    features, responses, weights, fold_assignments = training_data.build_regression_training_data_ordered(data_config)
else:
    data_management = load_training_data(data_config)


network_config = NetworkConfig(inshape=data_config.feature_shape[1:],
                               internal_window_radius=data_config.internal_window_radius,
                               **global_options)




cnn = Experiment(network_config, data_config, reinitialize=True)


# TODO add option for plotting training data previews

if (args.key == 'train' or args.key == 'all'):


feature_scaler = rsCNN.data_management.transforms.RobustTransformer(
    data_config.feature_nodata_value, data_config.data_save_name + '_feature_')
response_scaler = rsCNN.data_management.transforms.StandardTransformer(
    data_config.response_nodata_value, data_config.data_save_name + '_response_')

    train_set = fold_assignments == network_config.verification_fold
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

