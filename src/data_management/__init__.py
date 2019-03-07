

class Data_Config:
    """ A wrapper class designed to hold all relevant information about data sources,
        sample generation, and scaling.
    """

    def __init__(self, window_radius, feature_file_list, response_file_list):
        """
          Arguments:
          window_radius - determines the subset image size, which results as 2*window_radius  
          feature_file_list - file list of the feature rasters
          response_file_list - file list of the response rasters
        """

        # determines the subset image size, which results as 2*window_radius
        self.window_radius = window_radius

        # file list of the feature rasters
        self.feature_file_list = self.feature_file_list

        # file list of the response rasters
        self.response_file_list = self.response_file_list

        #####################################################################################
        ################# set defaults (to be overridden later)  ############################
        #####################################################################################

        # A boolean indication of whether the response type is a vector or a raster (True for vector).
        self.response_as_vectors = False

        # An optional list of boundary files for each feature/response file.
        self.boundary_file_list = []

        # A boolean indication of whether the boundary file type is a vector or a raster (True for vector).
        self.boundary_as_vectors = False

        # Value that indicates pixels that are 'out of bounds' in a boundary raster file
        self.boundary_bad_value = 0

        # An inner image subset used to score the algorithm, and within which a response must lie to
        # be included in training data
        self.internal_window_radius = window_radius

        # The fraction to randomly shuffle data from around response center.
        self.center_random_offset_fraction = 0

        # The maximum fraction of nodata_values to allow in each training sample.
        self.nodata_maximum_fraction = 0
        # A flag to fill in missing data with a nearest neighbor interpolation.
        self.fill_in_feature_data = False

        # TODO: rename/revise these as appropriate
        self.global_scaling = None
        self.local_scaling = None

        # A random seed to set (for reproducability), set to None to not set a seed.
        self.random_seed = 13

        # The number of folds to set up for data training.
        self.n_folds = 10

        # Either an integer (used for all sites) or a list of integers (one per site)
        # that designates the maximum number of samples to be pulled
        # from that location.  If the number of responses is less than the samples
        # per site, than the number of responses available is used
        self.max_samples = 1e12

        # A flag to ignore projection differences between feature and response sets - use only if you
        # are sure the projections are really the same.
        self.ignore_projections = False

        #  The value to ignore from the feature or response dataset.
        self.feature_nodata_value = -9999
        self.response_nodata_value = -9999

        # The minimum and maximum values for the resopnse dataset
        self.response_min_value = None
        self.response_max_value = None

        # Sampling type - options are 'ordered' and 'bootstrap'
        self.sample_type = 'ordered'

        # if None, don't save the data name, otherwise, do save requisite components as npz files
        # based on this root extension
        self.data_save_name = None

        # stored values for the eventual feature and response shapes
        self.response_shape = None
        self.feature_shape = None

    # TODO
    # def read_from_file(filename):
    #  """ Read in optional arguments from file
    #  Arguments:
    #  filename - str
    #    Name of file to read.
    #  """

    def read_from_dict(in_dict):
        """ Read in optional arguments from dictionary
        Arguments:
        in_dict - dictionary
          Dictionary of values to read in.
        """
        for key in in_dict:
            if (key == 'response_as_vectors'):
                self.response_as_vectors = in_dict[key]
            elif (key == 'boundary_file_list'):
                self.boundary_file_list = in_dict[key]
            elif (key == 'boundary_as_vectors'):
                self.boundary_as_vectors = in_dict[key]
            elif (key == 'boundary_bad_value'):
                self.boundary_bad_value = in_dict[key]
            elif (key == 'internal_window_radius'):
                self.internal_window_radius = in_dict[key]
            elif (key == 'center_random_offset_fraction'):
                self.center_random_offset_fraction = in_dict[key]
            elif (key == 'nodata_maximum_fraction'):
                self.nodata_maximum_fraction = in_dict[key]
            elif (key == 'fill_in_feature_data'):
                self.fill_in_feature_data = in_dict[key]
            elif (key == 'global_scaling'):
                self.global_scaling = in_dict[key]
            elif (key == 'local_scaling'):
                self.local_scaling = in_dict[key]
            elif (key == 'random_seed'):
                self.random_seed = in_dict[key]
            elif (key == 'n_folds'):
                self.n_folds = in_dict[key]
            elif (key == 'max_samples'):
                self.max_samples = in_dict[key]
            elif (key == 'ignore_projections'):
                self.ignore_projections = in_dict[key]
            elif (key == 'feature_nodata_value'):
                self.feature_nodata_value = in_dict[key]
            elif (key == 'response_nodata_value'):
                self.response_nodata_value = in_dict[key]
            elif (key == 'response_min_value'):
                self.response_min_value = in_dict[key]
            elif (key == 'response_max_value'):
                self.response_max_value = in_dict[key]
            elif (key == 'sample_type'):
                self.sample_type = in_dict[key]
            elif (key == 'data_save_name'):
                self.data_save_name = in_dict[key]
