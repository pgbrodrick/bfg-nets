import os


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
        # TODO:  Phil:  this is the only instance of this in the repository, remove? If not, then is there a way we
        #  can detect this automatically?
        self.response_as_vectors = kwargs.get('response_as_vectors', False)

        # An optional list of boundary files for each feature/response file.
        self.boundary_file_list = kwargs.get('boundary_file_list', [])

        # A boolean indication of whether the boundary file type is a vector or a raster (True for vector).
        # TODO:  Phil:  this is used in one place, can we detect this automatically?
        self.boundary_as_vectors = kwargs.get('boundary_as_vectors', False)

        # Value that indicates pixels that are 'out of bounds' in a boundary raster file
        self.boundary_bad_value = kwargs.get('boundary_bad_value', 0)

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

        # The minimum and maximum values for the response dataset
        # TODO:  Phil:  is this the raw min and max values or is this the min and max after the algorithm cleans
        #  the data (e.g., by truncating data?)
        self.response_min_value = kwargs.get('response_min_value', None)
        self.response_max_value = kwargs.get('response_max_value', None)

        # Sampling type - options are 'ordered' and 'bootstrap'
        # TODO:  Phil:  this is the only place this is used, remove? Alternatively, if this is the same "ordered" as
        #  in "ordered_continuous" and "ordered_categorical" above, ordered makes more sense in this context. It might
        #  be worth keeping them as separate options so that it's easier to do different pairwise combinations and
        #  introduce new sampling methods in the future.
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
        self.response_scaler_name = kwargs.get('response_scaler_name', 'NullScaler')
        self.feature_mean_centering = kwargs.get('feature_mean_centering', False)

        self.apply_random_transformations = kwargs.get('apply_random_transformations', False)

        self.features = None
        self.responses = None
        self.weights = None

        self.feature_scaler = None
        self.response_scaler = None
