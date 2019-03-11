import sys
import gdal

# TODO manage imports
from src.util.general import *
from src.networks import CNN, NetworkConfig
from src.data_management import Data_Config, training_data, transforms
fr

# TODO script needs to be adapted yet

key = sys.argv[1]

window_radius = 16

year = '2015'
feature_files = ['dat/features/feat_subset.tif']
response_files = ['dat/responses/resp_subset.tif']

# TODO: if we want to grab these from a config file, need to write a wrapper to read the transform in
data_options = {
    'data_save_name': 'munged/cnn_munge_' + str(window_radius) + '_test',
    'internal_window_radius': rint(window_radius*0.5),
    'max_samples': 30000,
    'max_value': 10000,
    'min_value': 0,
}

data_config = Data_Config(window_radius, feature_files, response_files, **data_options)


network_options: {,
                  'batch_norm': False,
                  'conv_depth': 16,
                  'conv_pattern': 3,
                  'n_layers': 10,
                  'network_name': 'cwc_test_network',
                  'output_activation': 'softplus',

                  'batch_size': 100,
                  'max_epochs': 100,
                  'n_noimprovement_repeats': 30,
                  'output_directory': None,
                  'verification_fold': 0,
                  }

network_config = NetworkConfig('flat_regress_net',
                               config.feature_shape[1:],
                               n_classes=1,
                               **network_options)

cnn = CNN(network_config)


if (key == 'build' or key == 'all'):
    features, responses, fold_assignments = training_data.build_regression_training_data_ordered(data_config)


# TODO add option for plotting training data previews

if (key == 'train' or key == 'all'):

    feature_scaler = RobustTransformer(data_config.data_save_name + '_feature_')
    response_scaler = StandardTransformer(data_config.data_save_name + '_response_')

    train_set = fold_assignments == data_config.verification_fold
    feature_scaler.fit(features[train_set, ...])
    response_scaler.fit(responses[train_set, ...])

    cnn.fit(feature_scaler.transform(features), response_scaler.transform(responses), fold_assignments)


if (key == 'apply' or key == 'all'):
    a = None
    # :TODO finish
