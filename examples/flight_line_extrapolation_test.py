import gdal
import ogr
import rasterio.features
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#TODO manage imports
import build_training_data
import plot_utility
import train_model
import apply_model
import sys



#TODO script needs to be adapted yet

key = sys.argv[1]

window_radius=16
internal_window_radius=int(round(window_radius*0.5))

year = '2015'
feature_files = ['dat/features/feat_subset.tif']
response_files = ['dat/responses/resp_subset.tif']
munge_file = 'munged/cnn_munge_' + str(window_radius) + '_test'


max_samples = 30000
model_name = 'test_flex'
global_scaling=None
local_scaling=None


if (key == 'build' or key == 'all'):
  features,responses,fold_assignments = build_training_data.\
                                      build_regression_training_data_ordered(\
                                      window_radius,
                                      max_samples,
                                      feature_files,
                                      response_files,
                                      internal_window_radius=internal_window_radius,
                                      fill_in_feature_data=True,
                                      local_scale_flag=local_scaling,
                                      global_scale_flag=global_scaling,
                                      nodata_maximum_fraction=0.0,
                                      ignore_projections=True,
                                      savename=munge_file,
                                      response_min_value = 0,
                                      response_max_value=10000,
                                      verbose=2)


if (key == 'train'):
  npzf = np.load(munge_file + '.npz')
  features = npzf['features']
  responses = npzf['responses']
  fold_assignments = npzf['fold_assignments']

if (key == 'plot'):
  npzf = np.load(munge_file + '.npz')
  features = npzf['features']
  responses = npzf['responses']
  fold_assignments = npzf['fold_assignments']
  plot_utility.plot_training_data(features,responses,images_to_plot=5,feature_band=2)
  plt.savefig('figs/trainig_data.png',dpi=200)

if (key == 'train' or key == 'all'):

  ## TODO: rememeber to add option for constant value scaling

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

