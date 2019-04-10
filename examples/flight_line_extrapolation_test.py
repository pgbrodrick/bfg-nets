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

from gdcs.utils import logging
logging.get_root_logger('debug_out.out')

# Initialize in one of three modes:
#   1) config file / settings if we want to
#   2) reloading from previous initialization
#   3) Sequence attachment + (i.e., other things needed)

#   sequence of numpy files


# Data build from raw files
#   1) 'Ordered crawl' - block-wise run through the image (out of core)
#   2) 'Response-centric crawl' - get our response, but keep our sample size reasonable
#   3) 'Feature-centric crawl' - think through making sure this is still enabled
#   4) 'Out of core bootstrap' - only grab images on the fly (randomly) --- table for now

# Partly a function of the above,
#   1) For CC, you need to convert response data inputs in CC format
#   2) Build weights as necessary

# Initialize scalers

# Construct (or attache pre-existing) sequences

# Grab some basic info from the first sequence elements (AKA, find inshape)



parser = argparse.ArgumentParser(description='Example spatial extrapolation application')
parser.add_argument('settings_file')
parser.add_argument('key')
args = parser.parse_args()

assert os.path.isfile(args.settings_file), 'Settings file: ' + args.settings_file + ' does not exist'
global_options = yaml.load(open(args.settings_file,'r'))

global_options['raw_feature_file_list'] = ['../global_cwc/dat/features/feat_subset.tif']
global_options['raw_response_file_list'] = ['../global_cwc/dat/responses/resp_subset.tif']


data_config = rsCNN.data_management.DataConfig(**global_options)
training_data.build_or_load_data(data_config)
training_data.build_or_load_scalers(data_config)

train_folds = [x for x in range(
    data_config.n_folds) if x is not data_config.validation_fold and x is not data_config.test_fold]

training_sequence = sequences.build_memmaped_sequence(data_config, train_folds, batch_size=100)
validation_sequence = sequences.build_memmaped_sequence(data_config, [data_config.validation_fold], batch_size=100)


#Move the inshape intot he build_or_load_model
print(global_options)
network_config = create_network_config(inshape=(data_config.window_radius*2,data_config.window_radius*2,3),**global_options)

experiment = Experiment(network_config, resume=True)
experiment.build_or_load_model()

if (args.key == 'train' or args.key == 'all'):
    experiment.fit_network(training_sequence, validation_sequence)

if (args.key == 'report' or args.key == 'all'):
    report = rsCNN.evaluation.ExperimentReport(experiment,
                                               validation_sequence)
                                               #training_sequence,
    report.create_report()


if (args.key == 'application'):
    application_feature_files = ['../global_cwc/dat/features/feat_subset.tif']
    application_output_basenames = ['examples/output/feat_subset_applied_cnn.tif']
# if (args.key == 'apply' or args.key == 'all'):
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


