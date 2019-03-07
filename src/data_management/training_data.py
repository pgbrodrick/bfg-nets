





import gdal
import numpy as np

from src.util.general import *



window_radius, max_samples, feature_file_list, response_file_list, boundary_file_list = [], boundary_as_vectors = True, boundary_bad_value=0, internal_window_radius=None, savename=None, nodata_maximum_fraction=0.5, fill_in_feature_data=False, global_scale_flag=None, local_scale_flag=None, nodata_value=-9999, random_seed=13, n_folds=10, verbose=False, ignore_projections=False, response_min_value=None, response_max_value=None


def build_regression_training_data_ordered(config):

  """ Builds a set of training data based on the configuration input
      Arguments:
      config - object of type Data_Config with the requisite values for preparing training data (see __init__.py)


      Return: 
      features - 4d numpy array 
        Array of data features, arranged as n,y,x,p, where n is the number of samples, y is the 
        data y dimension (2*window_radius), x is the data x dimension (2*window_radius), 
        and p is the number of features.
      responses - 4d numpy array
        Array of of data responses, arranged as n,y,x,p, where n is the number of samples, y is the 
        data y dimension (2*window_radius), x is the data x dimension (2*window_radius), 
        and p is the number of responses.  Each slice in the response dimension is a binary array o
        f that response class value.
      training_fold - numpy array 
        An array indicating which sample belongs to which data fold, from 0 to n_folds-1.
  """

  if (config.random_seed is not None):
    np.random.seed(config.random_seed)

  check_data_matches(config.feature_file_list,config.response_file_list,False,config.boundary_file_list,config.boundary_as_vectors,config.ignore_projections)

  if (isinstance(config.max_samples,list)):
    if (len(config.max_samples) != len(config.feature_file_list)):
      raise Exception('max_samples must equal feature_file_list length, or be an integer.')


  features = []
  responses = []
  repeat_index = []

  n_features = np.nan

  for _i in range(0,len(config.feature_file_list)):
    # TODO: external update through loggers
    
    # open requisite datasets
    dataset = gdal.Open(config.feature_file_list[_i],gdal.GA_ReadOnly)
    if (np.isnan(n_features)):
      n_features = dataset.RasterCount
    feature = np.zeros((dataset.RasterYSize,dataset.RasterXSize,dataset.RasterCount))
    for n in range(0,dataset.RasterCount):
      feature[:,:,n] = dataset.GetRasterBand(n+1).ReadAsArray()

    response = gdal.Open(config.response_file_list[_i]).ReadAsArray().astype(float)
    if (config.response_min_value is not None):
      response[response < config.response_min_value] = config.response_nodata_value
    if (response_max_value is not None):
      response[response > config.response_max_value] = config.response_nodata_value

    if (len(config.boundary_file_list) > 0):
      if (config.boundary_file_list[n] is not None):
        if (config.boundary_as_vectors):
          mask = rasterize_vector(config.boundary_file_list[_i],dataset.GetGeoTransform(),[feature.shape[0],feature.shape[1]])
        else:
          mask = gdal.Open(config.boundary_file_list[_i]).ReadAsArray().astype(float)
        feature[mask == config.boundary_bad_value,:] = config.feature_nodata_value
        response[mask == config.boundary_bad_value] = config.response_nodata_value


    # TODO: log feature shape

    # ensure nodata values are consistent 
    if (not dataset.GetRasterBand(1).GetNoDataValue() is None):
      feature[feature == dataset.GetRasterBand(1).GetNoDataValue()] = config.feature_nodata_value
    feature[np.isnan(feature)] = config.feature_nodata_value
    feature[np.isinf(feature)] = config.feature_nodata_value

    # TODO: consider whether we want the lines below included (or if we want them as an option)
    response[feature[:,:,0] == config.feature_nodata_value] = config.response_nodata_value
    feature[response == config.response_nodata_value,:] = config.feature_nodata_value

    pos_len=0
    cr = [0,feature.shape[1]]
    rr = [0,feature.shape[0]]
    
    collist = [x for x in range(cr[0]+config.window_radius,cr[1]-config.window_radius,config.internal_window_radius*2)]
    rowlist = [x for x in range(rr[0]+config.window_radius,rr[1]-config.window_radius,config.internal_window_radius*2)]

    colrow = np.zeros((len(collist)*len(rowlist),2)).astype(int)
    colrow[:,0] = np.matlib.repmat(np.array(collist).reshape((-1,1)),1,len(rowlist)).flatten()
    colrow[:,1] = np.matlib.repmat(np.array(rowlist).reshape((1,-1)),len(collist),1).flatten()
    del collist
    del rowlist

    colrow = colrow[np.random.permutation(colrow.shape[0]),:]

 
    # TODO: replace tqdm with log?
    for _cr in tqdm(range(len(colrow)),ncols=80):
      col = colrow[_cr,0]
      row = colrow[_cr,1]

      r = response[row-config.window_radius:row+config.window_radius,col-config.window_radius:col+config.window_radius].copy()
      if(r.shape[0] == config.window_radius*2 and r.shape[1] == config.window_radius*2):
        if ((np.sum(r == config.response_nodata_value) <= r.size * config.nodata_maximum_fraction)):
          d = feature[row-config.window_radius:row+config.window_radius,col-config.window_radius:col+config.window_radius].copy()

          responses.append(r)
          features.append(d)
          pos_len +=1

          if (pos_len >= max_samples):
            break
  
      
  # stack images up
  features = np.stack(features)
  responses = np.stack(responses)

  # randombly permute data to reshuffle everything
  perm = np.random.permutation(features.shape[0])
  features = features[perm,:]
  responses = responses[perm,:]
  fold_assignments = np.zeros(responses.shape[0])

  for f in range(0,config.n_folds):
    fold_assignments[rint(float(f)/float(n_folds)*len(fold_assignments)):rint(float(f+1)/float(n_folds)*len(fold_assignments))] = f
    
  # reshape images for the CNN
  features = features.reshape((features.shape[0],features.shape[1],features.shape[2],n_features))
  responses = responses.reshape((responses.shape[0],responses.shape[1],responses.shape[2],1))

  # TODO: convert below to log
  if(verbose > 0): print(('feature shape',features.shape))
  if(verbose > 0): print(('response shape',responses.shape))
  
  if (config.savename is not None):
    np.savez(config.savename,features=features,responses=responses,fold_assignments=fold_assignments)

  config.response_shape = responses.shape
  config.feature_shape = features.shape
  
  return features,responses,fold_assignments




