import gdal
import os
from rsCNN.data_management import scalers


class DataConfig:
    """ A wrapper class designed to hold all relevant information about data sources,
        sample generation, and scaling.
    """

    def __init__(self, **kwargs):

        # TODO: put appropriate asserts for the below values in build_training_data
        # TODO:  I think we'll want to structure and organize this when the config parameters are relatively stable,
        #  aren't changing much
        """
          Arguments:
          window_radius - determines the subset image size, which results as 2*window_radius
          feature_file_list - file list of the feature rasters
          response_file_list - file list of the response rasters
        """

        # determines the subset image size, which results as 2*window_radius
        self.window_radius = kwargs.get('window_radius', None)

        # file list of the feature rasters
        self.raw_feature_file_list = kwargs.get('raw_feature_file_list', [])

        # file list of the response rasters
        self.raw_response_file_list = kwargs.get('raw_response_file_list', [])

        # An optional list of boundary files for each feature/response file.
        self.boundary_file_list = kwargs.get('raw_boundary_file_list', [])

        self.check_input_files(self.raw_feature_file_list, self.raw_response_file_list, self.boundary_file_list)

        # Data type from each feature.  R == Real, C == categorical
        # All Categorical bands will be one-hot-encoded...to keep them as 
        # a single band, simply name them as real datatypes
        self.feature_raw_band_types, = get_band_types(self.raw_feature_file_list, kwargs.get('response_raw_band_types',None))
        self.response_raw_band_types, = get_band_types(self.raw_response_file_list, kwargs.get('response_raw_band_types',None))

        # A boolean indication of whether the boundary file type is a vector or a raster (True for vector).
        self.boundary_as_vectors = [os.path.splitext(x) == 'kml' or os.path.splitext(x) == 'shp' for x in self.boundary_file_list]

        # Value that indicates pixels that are 'out of bounds' in a boundary raster file
        self.boundary_bad_value = kwargs.get('boundary_bad_value', 0)





        # A string that tells us how to build the training data set.  Current options are:
        # ordered_continuous
        # TODO:  Phil:
        #  - ordered doesn't mean much to me without any context, can we make the name clearer either via documentation
        #    or, preferably, in how it's used?
        #  - I don't think this should have a default value, I think this should fail out if it's not provided and
        #    the data needs to be built
        #  - another option is to have functions called build_or_load_data_categorical and
        #    build_or_load_data_continuous... I don't think we want to do this necessarily, but it's an example of an
        #    option where the user is explicitly forced to make a decision about how the data is built without needing
        #    to know the correct config option
        self.data_build_category = kwargs.get('data_build_category', 'ordered_continuous')

        # A boolean indication of whether the response type is a vector or a raster (True for vector).
        # TODO:  Implement option
        #self.response_as_vectors = kwargs.get('response_as_vectors', False)

        # An inner image subset used to score the algorithm, and within which a response must lie to
        # be included in training data
        self.internal_window_radius = kwargs.get('internal_window_radius', self.window_radius)

        # The fraction to randomly shuffle data from around response center.
        # TODO:  Phil:  what does this do?
        self.center_random_offset_fraction = kwargs.get('center_random_offset_fraction', 0)

        # The maximum fraction of nodata_values to allow in each training sample.
        self.nodata_maximum_fraction = kwargs.get('nodata_maximum_fraction', 0)

        # A flag to fill in missing data with a nearest neighbor interpolation.
        self.fill_in_feature_data = kwargs.get('fill_in_feature_data', False)

        # A random seed to set (for reproducability), set to None to not set a seed.
        self.random_seed = kwargs.get('random_seed', 13)

        # The number of folds to set up for data training.
        self.n_folds = kwargs.get('n_folds', 10)

        # The fold to use for validation in model training
        self.validation_fold = kwargs.get('validation_fold', None)

        # The fold to use for testing
        self.test_fold = kwargs.get('test_fold', None)

        # Either an integer (used for all sites) or a list of integers (one per site)
        # that designates the maximum number of samples to be pulled
        # from that location.  If the number of responses is less than the samples
        # per site, than the number of responses available is used
        self.max_samples = kwargs.get('max_samples', 1e12)

        # A flag to ignore projection differences between feature and response sets - use only if you
        # are sure the projections are really the same.
        self.ignore_projections = kwargs.get('ignore_projections', False)

        #  The value to ignore from the feature or response dataset.
        self.feature_nodata_value = kwargs.get('feature_nodata_value', -9999)
        self.response_nodata_value = kwargs.get('response_nodata_value', -9999)
        self.feature_training_nodata_value = kwargs.get('feature_training_nodata_value', -10)

        # The minimum and maximum values for the response dataset
        # TODO:  Phil:  is this the raw min and max values or is this the min and max after the algorithm cleans
        #  the data (e.g., by truncating data?)
        self.response_min_value = kwargs.get('response_min_value', None)
        self.response_max_value = kwargs.get('response_max_value', None)

        # Sampling type - options are 'ordered' and 'bootstrap'
        self.sample_type = kwargs.get('sample_type', 'ordered')

        # if None, don't save the data name, otherwise, do save requisite components as npz files
        # based on this root extension
        # TODO:  Phil:  I'm not quite sure I'm on board with data_save_name as a parameter. I think concepts like
        #   directories and filenames and filepaths make sense intuitively, but data_save_name could be any of those
        #   objects. Is there a compelling reason to ever stick built data files in the same directory with one another?
        #   i.e., is there a reason to do this directory structure:
        #     dir_data
        #     |-- build1_features.npy
        #     |-- build1_responses.npy
        #     |-- build1_success
        #     |-- build2_features.npy
        #     |-- build2_responses.npy
        #     |-- build2_success
        #     |-- build3_features.npy
        #     |-- build3_responses.npy
        #     |-- build3_success
        #   when you could do this instead:
        #     dir_data
        #     |-- build1
        #     |  |-- features.npy
        #     |  |-- responses.npy
        #     |  |-- success
        #     |-- build2
        #     |  |-- features.npy
        #     |  |-- responses.npy
        #     |  |-- success
        #     |-- build3
        #     |  |-- features.npy
        #     |  |-- responses.npy
        #     |  |-- success
        #   If there isn't a compelling reason to accommodate files in the same directory, then we can make
        #   data_save_name much more explicit by calling it dir_build or something similar, and then just save to files
        #   in that directory without prepending anything
        self.data_save_name = kwargs.get('data_save_name', None)

        assert self.data_save_name is not None, 'current workflow requires a data save name'
        if (self.data_save_name is not None):
            # TODO:  Phil:  is there a reason we don't just make the directory?
            if not os.path.exists(os.path.dirname(self.data_save_name)):
                os.makedirs(os.path.dirname(self.data_save_name))
            #assert os.path.isdir(os.path.dirname(self.data_save_name)), 'Invalid path for data_save_name'
            self.response_files = [self.data_save_name + '_responses_' +
                                   str(fold) + '.npy' for fold in range(self.n_folds)]
            self.feature_files = [self.data_save_name + '_features_' +
                                  str(fold) + '.npy' for fold in range(self.n_folds)]
            self.weight_files = [self.data_save_name + '_weights_' + str(fold) + '.npy' for fold in range(self.n_folds)]
            # TODO:  Phil:  I don't think this should be a config parameter. I think you'd just want a hardcoded filename
            #  and then a single utility function that checks for that file. Also, I like your idea of saving the
            #  data config in the build data directory and then checking for that file's existence.
            self.successful_data_save_file = self.data_save_name + '_success'

        # Scalers
        # TODO:  Phil:  this is the real reason I started looking at this file. My model ran with no transformations
        #  because this module silently used a NullScaler when I had no scaler selected. I think we should make these
        #  required parameters.
        self.feature_scaler_name = kwargs.get('feature_scaler_name', 'NullScaler')
        if (type(self.feature_scaler_name) is list):
            assert len(self.feature_scaler_name) == len(self.raw_feature_file_list)
            for _s in range(len(self.feature_scaler_name)):
                scalers.check_scaler_exists(self.feature_scaler_name[_s])

        self.response_scaler_name = kwargs.get('response_scaler_name', 'NullScaler')
        if (type(self.response_scaler_name) is list):
            assert len(self.response_scaler_name) == len(self.raw_response_file_list)
            for _s in range(len(self.response_scaler_name)):
                scalers.check_scaler_exists(self.response_scaler_name[_s])

        # Data type from each feature.  R == Real, C == categorical
        # All Categorical bands will be one-hot-encoded...to keep them as 
        # a single band, simply name them as real datatypes
        self.feature_band_types = kwargs.get('feature_band_types','R')
        if (type(self.feature_band_types) is list):
            assert len(self.feature_band_types)

        self.feature_mean_centering = kwargs.get('feature_mean_centering', False)

        self.apply_random_transformations = kwargs.get('apply_random_transformations', False)

        self.features = None
        self.responses = None
        self.weights = None

        self.feature_scaler = None
        self.response_scaler = None

    def check_input_files(f_file_list, r_file_list, b_file_list):
        
        # f = feature, r = response, b = boundary

        # file lists r and f are expected a list of lists.  The outer list is a series of sites (location a, b, etc.).  
        # The inner list is a series of files associated with that site (band x, y, z).  Each site must have the
        # same number of files, and each file from each site must have the same number of bands, in the same order.
        # file list b is a list for each site, with one boundary file expected to be the interior boundary for all bands.

        # Check that feature and response files are lists
        assert type(f_file_list) is list, 'Feature files must be a list of lists'
        assert type(r_file_list) is list, 'Response files must be a list of lists'

        # Checks on the matching numbers of sites
        assert len(f_file_list) == len(r_file_list), 'Feature and response site lists must be the same length'
        assert len(f_file_list) > 0, 'At least one feature and response site is required'
        if (len(b_file_list) > 0):
            assert len(b_file_list) == len(f_file_list), 'Boundary and feature site lists must be the same length'

        # Checks that we have lists of lists for f and r
        for _f in range(len(f_file_list)):
            assert type(f_file_list[_f]) is list, 'Features at site {} are not as a list'.format(_f)
            assert type(r_file_list[_f]) is list, 'Responses at site {} are not as a list'.format(_f)

        # Checks that all files can be opened by gdal
        for _site in range(len(f_file_list)):
            assert type(f_file_list[_site]) is list, 'Features at site {} are not as a list'.format(_site)
            assert type(r_file_list[_site]) is list, 'Responses at site {} are not as a list'.format(_site)
            for _band in range(len(f_file_list[_site])):
                assert gdal.Open(f_file_list[_site][_band],gdal.GA_ReadOnly) is not None, 
                       'Could not open feature site {}, file {}'.format(_site,_band)
            for _band in range(len(r_file_list[_site])):
                assert gdal.Open(r_file_list[_site][_band],gdal.GA_ReadOnly) is not None, 
                       'Could not open response site {}, file {}'.format(_site,_band)

        # Checks on the number of files per site
        num_f_files_per_site = len(f_file_list[0])
        num_r_files_per_site = len(r_file_list[0])
        for _site in range(len(f_file_list)):
            assert len(f_file_list[_site]) == num_f_files_per_site, 'Inconsistent number of feature files at site {}'.format(_site)
            assert len(r_file_list[_site]) == num_r_files_per_site, 'Inconsistent number of response files at site {}'.format(_site)

        # Checks on the number of bands per file
        num_f_bands_per_file = [gdal.Open(x,gdal.GA_ReadOnly).RasterCount for x in f_file_list[0]]
        num_r_bands_per_file = [gdal.Open(x,gdal.GA_ReadOnly).RasterCount for x in r_file_list[0]]
        for _site in range(len(f_file_list)):
            for _file in range(len(f_file_list[_site])):
                assert gdal.Open(f_file_list[_site][_file],gdal.GA_ReadOnly).RasterCount == num_f_bands_per_file[_file],
                       'Inconsistent number of feature bands in site {}, file {}'.format(_site,_band)

            for _file in range(len(r_file_list[_site])):
                assert gdal.Open(r_file_list[_site][_file],gdal.GA_ReadOnly).RasterCount == num_r_bands_per_site[_file]
                       'Inconsistent number of response bands in site {}, file {}'.format(_site,_band)


    def get_band_types(file_list, band_types):

        valid_band_types = ['R','C']
        # 3 options are available for specifying band_types:
        # 1) band_types is None - assume all bands are real
        # 2) band_types is a list of strings within valid_band_types - assume each band from the associated file is the specified type,
        #    requires len(band_types) == len(file_list[0])
        # 3) band_types is list of lists (of strings, contained in valid_band_types), with the outer list referring to 
        #    files and the inner list referring to bands

        num_bands_per_file = [gdal.Open(x,gdal.GA_ReadOnly).RasterCount for x in file_list[0]]

        # Nonetype, option 1 from above, auto-generate
        if (band_types is None):
            for _file in range(len(file_list[0])):
                output_raw_band_types = []
                output_raw_band_types.append(['R' for _band in range(num_bands_per_file[_file])])

        else:
            assert(type(band_types) is list, 'band_types must be None or a list')

            # List of lists, option 3 from above - just check components
            if (type(band_types[0]) is list):
                for _file in range(len(band_types)):
                    assert(type(band_types[_file]) is list, 'If one element of band_types is a list, all elements must be lists'
                    assert len(band_types[_file]) == num_bands_per_file[_file], 'File {} has wrong number of band types'.format(_file)
                    for _band in range(len(band_types[_file])):
                        assert band_types[_file][_band] in valid_band_types, 'Invalid band types at file {}, band {}'.format(_file,_band)
                
                output_raw_band_types = band_types

            else:
                # List of values valid_band_types, option 2 from above - convert to list of lists
                output_raw_band_types = []
                for _file in range(len(band_types)):
                    assert band_types[_file] in valid_band_types, 'Invalid band type at File {}'.format(_file)
                    output_raw_band_types.append([band_types[_file] for _band in range(num_bands_per_file[_file])])
            
        return output_raw_band_types









