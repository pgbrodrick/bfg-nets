import gdal
import numpy as np

from src.util.general import *



def apply_semantic_segmentation(input_file_list,\
                                output_folder,\
                                model,\
                                window_radius,\
                                application_name='',
                                internal_window_radius=None,\
                                make_png=True,\
                                make_tif=True,\
                                local_scale_flag=None,\
                                global_scale_file=None,\
                                png_dpi=200,\
                                verbose=False,
                                nodata_value=-9999):
    """ Apply a trained model to a series of files.
  
    Arguments:
    input_file_list - list
      List of feature files to apply the model to.
    output_folder - str
      Directory to place output images into.
    model - keras CNN model
      A pre-trained keras CNN model for semantic segmentation.
  
    Keyword Arguments:
    application_name - str
      A string to add into the output file name.
    internal_window_radius - int
      The size of the internal window on which to score the model.
    make_png - boolean
      Should an output be created in PNG format?
    make_tif - boolean
      Should an output be created in GeoTiff format?
    local_scale_flag - str
      A flag to apply local scaling (ie, scaling at the individual image level).
      Should match the local_scale_flage used to prepare training data.
      Options are:
        mean - mean center each image
        mean_std - mean center, and standard deviatio normalize each image
    global_scale_file - str
      A file to pull global scaling from.
    png_dpi - int
      The dpi of the generated PNG, if make_png is set to true.
    verbose - boolean
      An indication of whether or not to print outputs.
    nodata_value - float
      The value to set as the output nodata_value.
  
    Return:
    None, simply generates the sepcified output images.
    """
  
    feature_dim = gdal.Open(input_file_list[0],gdal.GA_ReadOnly).RasterCount
  
    if (os.path.isdir(output_folder) == False): os.mkdir(output_folder)
  
    if (internal_window_radius is None): internal_window_radius = window_radius
  
    for f in input_file_list:
      output_tif_file = os.path.join(output_folder,os.path.basename(f).split('.')[0] + '_' + application_name + '_prediction.tif')
      output_png_file = os.path.join(output_folder,os.path.basename(f).split('.')[0] + '_' + application_name + '_prediction.png')
      
      if(verbose): print(f)
      dataset = gdal.Open(f,gdal.GA_ReadOnly)
      feature = np.zeros((dataset.RasterYSize,dataset.RasterXSize,dataset.RasterCount))
      for n in range(0,dataset.RasterCount):
        feature[:,:,n] = dataset.GetRasterBand(n+1).ReadAsArray()
  
      if (not dataset.GetRasterBand(1).GetNoDataValue() is None):
        feature[feature == dataset.GetRasterBand(1).GetNoDataValue()] = nodata_value
      feature[np.isnan(feature)] = nodata_value
      feature[np.isinf(feature)] = nodata_value
  
  
      if (global_scale_file is not None):
       npzf = np.load(global_scale_file)
       feature_scaling=npzf['feature_scaling']
       for n in range(0,feature.shape[2]):
        gd = feature[:,:,n] != nodata_value
        #feature_scaling = global_scale(feature[:,:,n],global_scale_flag,nd=nodata_value)
        feature[gd,n] = feature[gd,n] - feature_scaling[n,0]
        feature[gd,n] = feature[gd,n] / feature_scaling[n,1]
        
      gd = np.all(feature != nodata_value,axis=-1)
      
      feature[gd,:] = feature[gd,:] / 1000.
      feature[np.logical_not(gd),:] = 0

  
      n_classes = model.predict(np.zeros((1,window_radius*2,window_radius*2,feature.shape[-1]))).shape[-1]
      output = np.zeros((feature.shape[0],feature.shape[1],n_classes))+ nodata_value
   
      cr = [0,feature.shape[1]]
      rr = [0,feature.shape[0]]
      
      collist = [x for x in range(cr[0]+window_radius,cr[1]-window_radius,internal_window_radius*2)]
      collist.append(cr[1]-window_radius)
      rowlist = [x for x in range(rr[0]+window_radius,rr[1]-window_radius,internal_window_radius*2)]
      rowlist.append(rr[1]-window_radius)
  
      for col in collist:
        if(verbose): print((col,cr[1]))
        images = []
        for n in rowlist:
          d = feature[n-window_radius:n+window_radius,col-window_radius:col+window_radius].copy()
          if(d.shape[0] == window_radius*2 and d.shape[1] == window_radius*2):
            d = scale_image(d,local_scale_flag)
            d = fill_nearest_neighbor(d)
            images.append(d)
        images = np.stack(images)
        images = images.reshape((images.shape[0],images.shape[1],images.shape[2],dataset.RasterCount))
      
        pred_y = model.predict(images)
        
        _i = 0
        for n in rowlist:
          #p = np.squeeze(pred_y[_i,...])
          p = pred_y[_i,...]
          if (internal_window_radius < window_radius):
            mm = rint(window_radius - internal_window_radius)
            p = p[mm:-mm,mm:-mm,:]
          output[n-internal_window_radius:n+internal_window_radius,col-internal_window_radius:col+internal_window_radius,:] = p
          _i += 1
          if (_i >= len(images)):
            break
      
      output[np.logical_not(gd),:] = nodata_value
      if (global_scale_file is not None):
        npzf = np.load(global_scale_file)
        response_scaling=npzf['response_scaling']

        print('response scaling')
        gd = feature[:,:,0] != nodata_value
        #output[gd,:] = output[gd,:]*response_scaling[0,1] + response_scaling[0,0]
      gd[output[...,0] == nodata_value] = False
      output[gd,:] = output[gd,:]*1000
      output[np.logical_not(gd),:] = nodata_value
      
      if (make_png):
      
        output[output == nodata_value] = np.nan
        feature[feature == nodata_value] = np.nan
        gs1 = gridspec.GridSpec(1,n_classes+1)
        for n in range(0,n_classes):
          ax = plt.subplot(gs1[0,n])
          im = plt.imshow(output[:,:,n],vmin=0,vmax=1)
          plt.axis('off')
  
        ax = plt.subplot(gs1[0,n_classes])
        im = plt.imshow(np.squeeze(feature[...,0]))
        plt.axis('off')
        plt.savefig(output_png_file,dpi=png_dpi,bbox_inches='tight')
        plt.clf()
        
      if(verbose): print(output.shape) 
      if (make_tif):
        driver = gdal.GetDriverByName('GTiff') 
        driver.Register()
        output[np.isnan(output)] = nodata_value
         
        outDataset = driver.Create(output_tif_file,output.shape[1],output.shape[0],n_classes,gdal.GDT_Float32)
        outDataset.SetProjection(dataset.GetProjection())
        outDataset.SetGeoTransform(dataset.GetGeoTransform())
        for n in range(0,n_classes):
          if(verbose): print(np.squeeze(output[:,:,n]).shape)
          outDataset.GetRasterBand(n+1).WriteArray(np.squeeze(output[:,:,n]),0,0)
        del outDataset
      del dataset
 

