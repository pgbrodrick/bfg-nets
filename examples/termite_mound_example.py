


import os
import requests

from rsCNN.configuration import configs
from rsCNN.data_management import data_core, apply_model_to_data
import rsCNN.reporting.reports
from rsCNN.experiments import experiments
from rsCNN.utils import logging


DATA_DIR = 'scratch'

dem_data = ['https://github.com/pgbrodrick/ecoCNN/blob/master/data/landscape7_dem_subset.tif?raw=true', os.path.join(DATA_DIR, 'dem.tif')]
boundary_data = ['https://github.com/pgbrodrick/ecoCNN/raw/master/data/training_boundary_utm.shp?raw=true', os.path.join(DATA_DIR, 'training_boundary.geojson')]
mound_data = ['https://github.com/pgbrodrick/ecoCNN/raw/master/data/mound_points_utm.geojson?raw=true', os.path.join(DATA_DIR, 'mound_points_utm.geojson')]

def get_data_from_url(url, destination):
    if (os.path.isfile(destination) is False):
        r = requests.get(url)
        with open(destination, 'wb') as outfile:
            outfile.write(r.content)



get_data_from_url(dem_data[0], dem_data[1])
get_data_from_url(boundary_data[0], boundary_data[1])
get_data_from_url(mound_data[0], mound_data[1])

config = configs.create_config_from_file('settings_termite_mound_example.yaml')

data_container = data_core.DataContainer(config)

data_container.build_or_load_rawfile_data()
data_container.build_or_load_scalers()
data_container.load_sequences()

experiment = experiments.Experiment(config)
experiment.build_or_load_model(data_container=data_container)

experiment.fit_model_with_data_container(data_container, resume_training=True)

final_report = rsCNN.reporting.reports.Reporter(data_container, experiment, config)

application_feature_files = config.raw_files.feature_files[0]
application_output_basenames = ['examples/output/feat_subset_applied_cnn.tif']
for _f in range(len(application_feature_files)):
    rsCNN.data_management.apply_model_to_data.apply_model_to_raster(experiment.model,
                                                                    data_container,
                                                                    application_feature_files[_f],
                                                                    application_output_basenames[_f],
                                                                    make_png=False,
                                                                    make_tif=True)
