

import argparse
import os
import subprocess
import sys

import gdal
import requests

import bfgn.reporting.reports
from bfgn.configuration import configs
from bfgn.data_management import apply_model_to_data, data_core
from bfgn.experiments import experiments
from bfgn.utils import logging

parser = argparse.ArgumentParser(description='Example termite mound classification')
parser.add_argument('key', type=str, choices=['all', 'prelim_report', 'train', 'report', 'apply'])
parser.add_argument('-debug_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARN', 'ERROR'])
parser.add_argument('-debug_file', type=str, default=None)
parser.add_argument('-run_as_raster', type=bool, default=False)
args = parser.parse_args()

if args.debug_file is not None:
    logger = logging.get_bfgn_logger(None, args.debug_level, args.debug_file)

######################### Download sample data from ecoCNN ################################
DATA_DIR = 'scratch'
if (os.path.isdir(DATA_DIR) is False):
    os.mkdir(DATA_DIR)

dem_data = ['https://github.com/pgbrodrick/ecoCNN/blob/master/data/landscape7_dem_subset.tif',
            os.path.join(DATA_DIR, 'dem.tif')]
boundary_data = ['https://github.com/pgbrodrick/ecoCNN/raw/master/data/training_boundary_utm.shp',
                 os.path.join(DATA_DIR, 'training_boundary.shp')]
mound_data = ['https://github.com/pgbrodrick/ecoCNN/raw/master/data/mounds_utm.shp',
              os.path.join(DATA_DIR, 'mounds_utm.shp')]

shp_associates = ['.shp', '.shx', '.dbf', '.prj']


def get_data_from_url(url, destination):
    print('fetching {}'.format(destination))
    if (os.path.isfile(destination) is False):
        r = requests.get(url + '?raw=true')
        with open(destination, 'wb') as outfile:
            outfile.write(r.content)


get_data_from_url(dem_data[0], dem_data[1])
for ext in shp_associates:
    get_data_from_url(os.path.splitext(boundary_data[0])[0] + ext, os.path.splitext(boundary_data[1])[0] + ext)
    get_data_from_url(os.path.splitext(mound_data[0])[0] + ext, os.path.splitext(mound_data[1])[0] + ext)


if (args.run_as_raster):
    # Convert shapefiles to rasters and modify config to test other input types.
    feature_set = gdal.Open(dem_data[1], gdal.GA_ReadOnly)
    trans = feature_set.GetGeoTransform()
    cmd_str = 'gdal_rasterize ' + mound_data[1] + ' ' + os.path.splitext(mound_data[1])[0] +\
              '.tif -init 0 -burn 1 -te ' + str(trans[0]) + ' ' + str(trans[3] + trans[5]*feature_set.RasterYSize) +\
              ' ' + str(trans[0] + trans[1]*feature_set.RasterXSize) + ' ' + str(trans[3]) + ' -tr ' +\
              str(trans[1]) + ' ' + str(trans[5])
    subprocess.call(cmd_str, shell=True)

    cmd_str = 'gdal_rasterize ' + boundary_data[1] + ' ' + os.path.splitext(boundary_data[1])[0] +\
              '.tif -init 0 -burn 1 -te ' + str(trans[0]) + ' ' + str(trans[3] + trans[5]*feature_set.RasterYSize) +\
              ' ' + str(trans[0] + trans[1]*feature_set.RasterXSize) + ' ' + str(trans[3]) + ' -tr ' +\
              str(trans[1]) + ' ' + str(trans[5])
    subprocess.call(cmd_str, shell=True)


config = configs.create_config_from_file(os.path.join(
    os.path.dirname(sys.argv[0]), 'settings_termite_mound_example.yaml'))
if (args.run_as_raster):
    config.raw_files.boundary_files = [os.path.splitext(x)[0] + '.tif' for x in config.raw_files.boundary_files]
    config.raw_files.response_files = [[os.path.splitext(x)[0] + '.tif' for x in config.raw_files.response_files[0]]]

data_container = data_core.DataContainer(config)

data_container.build_or_load_rawfile_data()
data_container.build_or_load_scalers()
data_container.load_sequences()


experiment = experiments.Experiment(config)
experiment.build_or_load_model(data_container=data_container)

if (args.key == 'prelim_report' or args.key == 'all'):
    prelim_report = bfgn.reporting.reports.Reporter(data_container, experiment, config)
    prelim_report.create_model_report()

if (args.key == 'train' or args.key == 'all'):
    experiment.fit_model_with_data_container(data_container, resume_training=True)

if (args.key == 'report' or args.key == 'all'):
    final_report = bfgn.reporting.reports.Reporter(data_container, experiment, config)
    final_report.create_model_report()

if (args.key == 'apply' or args.key == 'all'):
    application_feature_files = config.raw_files.feature_files[0]
    application_output_basenames = ['scratch/applied_output_model']
    for _f in range(len(application_feature_files)):
        bfgn.data_management.apply_model_to_data.apply_model_to_site(experiment.model,
                                                                     data_container,
                                                                     [application_feature_files[_f]],
                                                                     application_output_basenames[_f],
                                                                     output_format="GTiff")
