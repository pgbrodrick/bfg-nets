import os
import sys
import gdal

# TODO manage imports
from rsCNN.utils.general import *
from rsCNN.networks import CNN, NetworkConfig, losses
from rsCNN.data_management import DataConfig, training_data, transforms, apply_model_to_data


# TODO script needs to be adapted yet

key = sys.argv[1]

window_radius = 16

year = '2015'
feature_files = ['../global_cwc/dat/features/feat_subset.tif']
response_files = ['../global_cwc/dat/responses/resp_subset.tif']

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


if (key == 'build' or key == 'all'):
    features, responses, fold_assignments = training_data.build_regression_training_data_ordered(data_config)
else:
    data_config = None  # TODO: load data_config from disk


network_options = {
    'batch_norm': False,
    'conv_depth': 16,
    'conv_pattern': [3],
    'n_layers': 10,
    'network_name': 'cwc_test_network',
    'output_activation': 'softplus',

    'batch_size': 100,
    'max_epochs': 10,
    'n_noimprovement_repeats': 30,
    'output_directory': None,
    'verification_fold': 0
}

loss_function = losses.cropped_loss('mae',features.shape[1],data_config.internal_window_radius*2)
network_config = NetworkConfig('flat_regress_net',
                               loss_function,
                               features.shape[1:],
                               n_classes=1,
                               **network_options)

cnn = CNN(network_config, reinitialize=True, load_history=False)


# TODO add option for plotting training data previews

if (key == 'train' or key == 'all'):

    feature_scaler = transforms.RobustTransformer(
        data_config.feature_nodata_value, data_config.data_save_name + '_feature_')
    response_scaler = transforms.StandardTransformer(
        data_config.response_nodata_value, data_config.data_save_name + '_response_')

    train_set = fold_assignments == network_config.verification_fold
    feature_scaler.fit_transform(features[train_set, ...])
    response_scaler.fit_transform(responses[train_set, ...,:-1])

    cnn.fit(features, responses, fold_assignments)


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
