import argparse
import os
import sys
import yaml

# TODO manage imports
from rsCNN.networks.experiments import Experiment
import rsCNN.data_management
from rsCNN.networks.network_configs import create_network_config
import rsCNN.data_management.apply_model_to_data
import rsCNN.evaluation
from rsCNN.data_management import training_data, sequences

from rsCNN.utils import logging

logger = logging.get_root_logger('debug_out.out')
_LOG_LEVEL = 'DEBUG'  # 'TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR'

logger.setLevel(_LOG_LEVEL)


parser = argparse.ArgumentParser(description='Example spatial extrapolation application')
parser.add_argument('settings_file')
parser.add_argument('key')
args = parser.parse_args()

assert os.path.isfile(args.settings_file), 'Settings file: ' + args.settings_file + ' does not exist'
global_options = yaml.load(open(args.settings_file, 'r'))

global_options['raw_feature_file_list'] = [['../global_cwc/dat/features/feat_subset.tif']]
global_options['raw_response_file_list'] = [['../global_cwc/dat/responses/resp_subset.tif']]


data_config = rsCNN.data_management.DataConfig(**global_options)

data_container = training_data.build_or_load_rawfile_data(data_config)
data_container.build_or_load_scalers()

training_sequence = sequences.build_memmapped_sequence(data_container, data_container.train_folds, batch_size=100)
validation_sequence = sequences.build_memmapped_sequence(
    data_container, [data_container.config.validation_fold], batch_size=100)

# Move the inshape intot he build_or_load_model
network_config = create_network_config(
    inshape=(data_config.window_radius*2, data_config.window_radius*2, 3), **global_options)

experiment = Experiment(network_config, resume=True)
experiment.build_or_load_model()

if (args.key == 'train' or args.key == 'all'):
    experiment.fit_network(training_sequence, validation_sequence)

if (args.key == 'report'):
    rsCNN.evaluation.create_model_report(experiment.model,
                                         experiment.network_config,
                                         training_sequence,
                                         validation_sequence,
                                         history=experiment.history)


if (args.key == 'apply' or args.key == 'all'):
    application_feature_files = ['../global_cwc/dat/features/feat_subset.tif']
    application_output_basenames = ['examples/output/feat_subset_applied_cnn.tif']
    for _f in range(len(application_feature_files)):
        rsCNN.data_management.apply_model_to_data.apply_model_to_raster(experiment.model,
                                                                        data_config,
                                                                        application_feature_files[_f],
                                                                        application_output_basenames[_f],
                                                                        make_png=False,
                                                                        make_tif=True,
                                                                        feature_transformer=data_container.feature_scaler,
                                                                        response_transformer=data_container.response_scaler)
