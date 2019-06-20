import argparse
import os

from rsCNN.configuration import configs
from rsCNN.data_management import data_core, apply_model_to_data
import rsCNN.evaluation
from rsCNN.experiments import experiments
from rsCNN.utils import logging

parser = argparse.ArgumentParser(description='Example spatial extrapolation application')
parser.add_argument('settings_file')
parser.add_argument('key', type=str, choices=['all', 'train', 'report', 'apply'])
parser.add_argument('-debug_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARN', 'ERROR'])
parser.add_argument('-debug_file', type=str, default='debug.out')
args = parser.parse_args()

logger = logging.get_root_logger(args.debug_file)
logger.setLevel(args.debug_level)

assert os.path.isfile(args.settings_file), 'Settings file: ' + args.settings_file + ' does not exist'
config = configs.create_config_from_file(args.settings_file)

data_container = data_core.DataContainer(config)

data_container.build_or_load_rawfile_data()
data_container.build_or_load_scalers()
data_container.load_sequences()

experiment = experiments.Experiment(config)
experiment.build_or_load_model(data_container=data_container)

if (args.key == 'prelim_report'):
    rsCNN.evaluation.create_preliminary_model_report_from_experiment(experiment, data_container)

if (args.key == 'train' or args.key == 'all'):
    experiment.fit_model_with_dataset(data_container, resume_training=True)

if (args.key == 'report' or args.key == 'all'):
    rsCNN.evaluation.create_model_report_from_experiment(experiment, data_container)

if (args.key == 'apply' or args.key == 'all'):
    application_feature_files = config.raw_files.feature_files[0]
    application_output_basenames = ['examples/output/feat_subset_applied_cnn.tif']
    for _f in range(len(application_feature_files)):
        rsCNN.data_management.apply_model_to_data.apply_model_to_raster(experiment.model,
                                                                        data_container,
                                                                        application_feature_files[_f],
                                                                        application_output_basenames[_f],
                                                                        make_png=False,
                                                                        make_tif=True)
