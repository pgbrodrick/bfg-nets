import sys

import gdal
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ogr
import rasterio.features

from src.data_management import Data_Config, training_data
from src.networks import CNN
from src.util.general import *
#TODO manage imports
import plot_utility
import train_model
import apply_model



#TODO script needs to be adapted yet

key = sys.argv[1]

window_radius=16

year = '2015'
feature_files = ['dat/features/feat_subset.tif']
response_files = ['dat/responses/resp_subset.tif']

data_config = Data_Config(window_radius, feature_files, response_files)
data_config.max_samples = 30000
data_config.data_save_name = 'munged/cnn_munge_' + str(window_radius) + '_test'
data_config.internal_window_radius=rint(window_radius*0.5)
data_config.global_scaling = None
data_config.local_scaling=None
data_config.min_value=0
data_config.max_value=10000


network_config = {}
network_config['conv_depth'] = 16
network_config['batch_norm'] = False
network_config['n_layers'] = 10
network_config['conv_pattern'] = 3
network_config['output_activation'] = 'softplus'
network_config['network_name'] = 'cwc_test_network'


########## TODO: Training values that need to be set somewhere
batch_size = 100
verification_fold = 0
n_noimprovement_repeats = 30

model_name = 'test_flex'


if (key == 'build' or key == 'all'):
  features,responses,fold_assignments = training_data.build_regression_training_data_ordered(data_config)

#TODO add option for plotting training data previews

if (key == 'train' or key == 'all'):

  ## TODO: rememeber to add option for constant value scaling

  cnn = CNN()
  cnn.create_config('flat_regress_net',config.feature_shape[1:],1,network_dictionary=network_config)


  # TODO: training needs to be done yet
  model = train_model.train_regression(features,
                                     responses,
                                     fold_assignments,
                                     model_name,
                                     internal_window_radius=internal_window_radius,
                                     network_name='flex_unet',
                                     verification_fold=0,
                                     batch_size=100,
                                     n_noimprovement_repeats=30,
                                     network_kwargs={'conv_depth':'growth','batch_norm':True,'output_activation':'softplus'})




if (key == 'apply' or key == 'all'):
  model = train_model.load_trained_model(model_name,window_radius,verbose=False,weighted=True,loss_type='mae')
  global_scale_file = munge_file + '_global_feature_scaling.npz'
  if (global_scaling == None): global_scale_file = None
  apply_model.apply_semantic_segmentation(feature_files,\
                              'output_maps',\
                              model,\
                              window_radius,\
                              internal_window_radius=internal_window_radius,\
                              local_scale_flag=local_scaling,\
                              global_scale_file=global_scale_file,\
                              make_png=False,\
                              make_tif=True,\
                              verbose=False,
                              nodata_value=-9999)

