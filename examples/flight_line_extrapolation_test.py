import os
import sys

# TODO manage imports
from rsCNN.utils.general import *
from rsCNN.networks import Experiment, NetworkConfig
from rsCNN.data_management import DataConfig, training_data, transforms, apply_model_to_data, load_training_data


# TODO script needs to be adapted yet

key = sys.argv[1]


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
    'batch_norm': False,
    'conv_depth': 16,
    'conv_pattern': [3],
    'n_layers': 10,
    'output_activation': 'softplus',
    'n_classes': 1,

    'network_name': 'cwc_test_network',
    'optimizer': 'adam',
    'batch_size': 100,
    'max_epochs': 200,
    'n_noimprovement_repeats': 30,
    'output_directory': None,
    'verification_fold': 0,
    'loss_metric': 'mae',

    'application_feature_files' : ['../global_cwc/dat/features/feat_subset.tif'],
    'application_output_basenames' : ['examples/output/feat_subset_applied_cnn.tif']
}



data_config = DataConfig( **global_options)

if (key == 'build' or key == 'all'):
    features, responses, weights, fold_assignments = training_data.build_regression_training_data_ordered(data_config)
else:
    data_management = load_training_data(data_config)


network_config = NetworkConfig(inshape=data_config.feature_shape[1:],
                               internal_window_radius=data_config.internal_window_radius,
                               **global_options)




cnn = Experiment(network_config, data_config, reinitialize=True)


# TODO add option for plotting training data previews

if (key == 'train' or key == 'all'):

    feature_scaler = transforms.RobustTransformer(
        data_config.feature_nodata_value, data_config.data_save_name + '_feature_')
    response_scaler = transforms.StandardTransformer(
        data_config.response_nodata_value, data_config.data_save_name + '_response_')

    print(np.mean(responses[responses[..., 0] != -9999, 0]))
    train_set = fold_assignments == network_config.verification_fold
    feature_scaler.fit(features[train_set, ...])
    response_scaler.fit(responses[train_set, ..., :-1])

    features = feature_scaler.transform(features)
    responses[..., :-1] = response_scaler.transform(responses[..., :-1])
    print(np.mean(responses[responses[..., 0] != -9999, 0]))

    cnn.fit(features, responses, fold_assignments, load_history=False)


if (key == 'apply' or key == 'all'):
    for _f in range(len(application_feature_files)):
        apply_model_to_data.apply_model_to_raster(cnn,
                                                  data_config,
                                                  application_feature_files[_f],
                                                  application_output_basenames[_f],
                                                  make_png=False,
                                                  make_tif=True,
                                                  feature_transformer=feature_scaler,
                                                  response_transformer=response_scaler)
