import argparse
import os
import sys
import gdal
import numpy as np

# TODO manage imports
from rsCNN.networks.model import Experiment
import rsCNN.data_management
from rsCNN.data_management import training_data
import rsCNN.data_management.transforms
import rsCNN.data_management.apply_model_to_data
import rsCNN.evaluation


# TODO script needs to be adapted yet

parser = argparse.ArgumentParser(description='CNN example for spatial extrapolation from CAO flight lines')
parser.add_argument('key')
args = parser.parse_args()


global_options = {
    'raw_feature_file_list': ['../global_cwc/dat/features/feat_subset.tif'],
    'raw_response_file_list': ['../global_cwc/dat/responses/resp_subset.tif'],

    'data_save_name': 'examples/munged/cwc_data_test',
    'data_build_category': 'ordered_continuous',
    'max_samples': 30000,
    'window_radius': 16,
    'internal_window_radius': 8,
    'response_max_value': 10000,
    'response_min_value': 0,

    'network_type': 'flat_regress_net',
    'use_batch_norm': True,
    'conv_depth': 'growth',
    'block_structure': (1, 1),
    'n_layers': 10,
    'output_activation': 'softplus',
    'n_classes': 1,
    'continue_training': False,

    'network_name': 'cwc_test_network',
    'optimizer': 'adam',
    'batch_size': 100,
    'max_epochs': 100,
    'n_noimprovement_repeats': 10,
    'output_directory': None,
    'verification_fold': 0,
    'loss_metric': 'mae',

    'application_feature_files': ['../global_cwc/dat/features/feat_subset.tif'],
    'application_output_basenames': ['examples/output/feat_subset_applied_cnn.tif']
}


data_config = DataConfig(**global_options)

network_config = NetworkConfig(inshape=data_config.feature_shape[1:],
                               internal_window_radius=data_config.internal_window_radius,
                               **global_options)


experiment = Experiment(network_config, data_config)
experiment.build_or_load_data()
experiment.fit_network()
#experiment.evaluate_network()


#if (args.key == 'apply' or args.key == 'all'):
#    feature_scaler.load()
#    response_scaler.load()
#    for _f in range(len(application_feature_files)):
#        rsCNN.data_management.apply_model_to_data.apply_model_to_raster(experiment,
#                                                                        data_config,
#                                                                        application_feature_files[_f],
#                                                                        application_output_basenames[_f],
#                                                                        make_png=False,
#                                                                        make_tif=True,
#                                                                        feature_transformer=feature_scaler,
#                                                                        response_transformer=response_scaler)
