import json
import os
import random

import numpy as np

from rsCNN.utils import logging


# Data parameters
VALUE_MISSING = 0
MAX_PROP_MISSING = 0.25


_logger = logging.get_child_logger(__name__)


# TODO:  this is all a holdover from my other code, should be hacked to pieces or maybe even discarded to match with
# TODO:  Phil's code. Also, perhaps we want to deemphasize "timesteps" in my code, since that's specific to my problem
# TODO:  domain, and instead refer to channels where those channels could be RGB bands, timesteps, hyperspectral, other
# TODO:  environmental data... Still. I think we'll find it useful to keep the idea of getting samples from rasters,
# TODO:  numpy data files, etc.


class BaseSampler(object):

    def get_sample(self, image_dim, num_timesteps):
        raise NotImplementedError

    def set_new_epoch(self):
        pass


class DataArraySampler(BaseSampler):
    data_array = None

    def __init__(self, data_array):
        self.data_array = data_array

    def get_sample(self, image_dim, num_timesteps):
        # TODO:  this should be updated to allow for incomplete time series data
        # Start index needs to allocate enough timesteps for time window and for the last, predicted image
        idx_time_start = random.randint(0, self.data_array.shape[0] - num_timesteps - 1)
        idx_width_start = random.randint(0, self.data_array.shape[1] - image_dim)
        idx_height_start = random.randint(0, self.data_array.shape[2] - image_dim)
        # Assemble indices for clarity
        idxs_time = np.arange(idx_time_start, idx_time_start + num_timesteps + 2)
        idxs_width = np.arange(idx_width_start, idx_width_start + image_dim + 1)
        idxs_height = np.arange(idx_height_start, idx_height_start + image_dim + 1)
        # Get actual data
        x_y = self.data_array[idxs_time, idxs_width, idxs_height, :]
        # Try again if the image has too much missing data
        prop_missing = np.sum((x_y == VALUE_MISSING).any(axis=2)) / (x_y.shape[0] * x_y.shape[1])
        if prop_missing >= MAX_PROP_MISSING:
            return self.get_sample(image_dim, num_timesteps)
        return x_y[:-1, :, :, :], x_y[-1:, :, :, :]  # Split into x and y


class NumpyFileSampler(BaseSampler):
    filepaths = None
    _filepaths_not_used_this_batch = None

    def __init__(self, filepaths):
        self.filepaths = filepaths
        self._filepaths_not_used_this_batch = self.filepaths.copy()

    def get_sample(self, image_dim, num_timesteps):
        _logger.trace('Get sample via numpy data file sampler')
        if len(self._filepaths_not_used_this_batch) == 0:
            _logger.warn('Batch size is greater than number of numpy data files, reusing data files')
            self.set_new_epoch()
        idx_sample = random.randint(0, len(self._filepaths_not_used_this_batch) - 1)
        filepath = self._filepaths_not_used_this_batch.pop(idx_sample)
        _logger.trace('Load sample from {}'.format(filepath))
        inputs, targets = np.load(filepath)['arr_0']  # arr_0 = default key
        assert inputs.shape[1] == image_dim and inputs.shape[2] == image_dim, \
            'Sample image dim ({}) does not match requested image dim ({})' \
                .format(list(inputs.shape)[1:3], [image_dim, image_dim])
        assert inputs.shape[0] >= num_timesteps, \
            'Sample timeseries length ({}) is less than requested timesteps ({})'.format(inputs.shape[0], num_timesteps)
        filepath_metadata = os.path.splitext(filepath)[0] + '.json'
        with open(filepath_metadata, 'r') as file_:
            sample = json.load(file_)
        sample['filepaths_rasters'] = sample['filepaths_rasters'][-num_timesteps:]
        sample['sample'] = (inputs[-num_timesteps:], targets)
        return sample

    def set_new_epoch(self):
        self._filepaths_not_used_this_batch = self.filepaths.copy()


class RasterFileSampler(BaseSampler):
    geospatial_index = None
    geospatial_sampler = None
    idxs_bands = None
    _cached_size = 10
    _cached_points = list()

    def __init__(self, geospatial_index, geospatial_sampler, idxs_bands):
        self.geospatial_index = geospatial_index
        self.geospatial_sampler = geospatial_sampler
        self.idxs_bands = idxs_bands

    def get_sample(self, image_dim, num_timesteps):
        _logger.trace('Get sample via raster file sampler')
        rasters = list()
        # Start by drawing points until we have one that overlaps with the available rasters
        # Note 1:  it's probably clear that you don't want to be sampling many points that are outside your rasters
        # Note 2:  we're assuming that a time series of any length is useful for sampling but set the minimum to two,
        # one for inputs and one for targets
        _logger.trace('Get random point with sufficient rasters')
        num_attempts = 0
        while len(rasters) < 2:
            num_attempts += 1
            point = self.geospatial_sampler.sample_points(1)[0]
            rasters = list(self.geospatial_index.intersection(point.bounds, objects='raw'))
        _logger.trace('Retrieved {} candidate rasters after {} attempts at point {}'
                      .format(len(rasters), num_attempts, point.coords[0]))
        rasters = sorted(rasters, key=lambda x: x.get('datetime'))
        raster_data, rasters_used = self._get_raster_timeseries(
            rasters, image_dim=image_dim, len_needed=num_timesteps+1, point=point)
        _logger.trace('Retrieved {} sufficient rasters'.format(len(raster_data)))
        if len(raster_data) < 2:
            _logger.trace('Insufficient timeseries raster data, repeat sampling process')
            return self.get_sample(image_dim, num_timesteps)
        # We may need to fill the input array with empty slices for missing time-series data
        num_missing = num_timesteps - len(raster_data) + 1
        data_missing = np.empty((num_missing, image_dim, image_dim, len(self.idxs_bands)))
        data_missing[:] = np.nan
        # TODO:  the following lines were inserted because I got an error and was unable to debug with the logs
        # TODO:  remove it when this is solved
        shapes = [list(rd.shape) for rd in raster_data]
        all_equal = all([shapes[0] == shape for shape in shapes])
        if not all_equal:
            _logger.warn('Shapes do not match for rasters -- shapes {} -- rasters {}'.format(shapes, rasters_used))
            return self.get_sample(image_dim, num_timesteps)
        shape_missing = list(data_missing.shape)[1:]
        if not shape_missing == shapes[0]:
            _logger.warn('Missing data shapes does not match raster shapes -- {} {}'.format(shape_missing, shapes[0]))
            return self.get_sample(image_dim, num_timesteps)
        return {
            'filepaths_rasters': rasters_used,
            'sample_location': point.coords[0],
            'sample': (np.concatenate([data_missing, np.array(raster_data[:-1])]), raster_data[-1]),
        }

    def _get_raster_timeseries(self, rasters, image_dim, len_needed, point):
        # We're trying to collect a randomly-selected time series while assuming that not all rasters have sufficient
        # data. We're fine with time series being shorter than the desired number of timesteps, but it needs to be at
        # least two images long so that we have an input and a target.
        # Ideally, the time series will have many complete rasters, so that we can randomly select a starting point for
        # the time series and get the correct number of images. Assuming we need X time steps in our time series, we
        # randomly select starting points that are at least X + 1 images before the end of the dataset. However, we may
        # find that one or more rasters are missing data and run out of candidate rasters before completing the time
        # series. In this case, we being moving backwards from the starting point until we find enough sufficient
        # rasters. There's still a chance that we reach the beginning of the time series before completing the time
        # series and, in this case, we return the time series as-is. We allow the higher-level functions to determine
        # what to do with this time series, but it's likely that we'll just continue the process until finding a
        # sufficient time series.
        idx_start = random.randint(0, len(rasters))
        counter_tested = 0
        counter_sufficient = 0
        rasters_used = list()
        raster_data = list()
        # Begin collecting rasters from the starting point, moving forward in time
        for raster in rasters[idx_start:]:
            if counter_sufficient == len_needed:
                break
            raster_datum = self._get_data_from_raster_file(raster['filepath'], image_dim, point)
            counter_tested += 1
            if raster_datum is not None:  # Not sufficient
                rasters_used.append(raster['filepath'])
                raster_data.append(raster_datum)  # Note that we append because the new raster comes later
                counter_sufficient += 1
        # Continue collecting rasters from the starting point, moving backward in time
        for raster in reversed(rasters[:idx_start]):
            if counter_sufficient == len_needed:
                break
            raster_datum = self._get_data_from_raster_file(raster['filepath'], image_dim, point)
            counter_tested += 1
            if raster_datum is not None:  # Not sufficient
                rasters_used.insert(0, raster['filepath'])
                raster_data.insert(0, raster_datum)  # Note that we insert because the new raster comes earlier
                counter_sufficient += 1
        _logger.trace('Rasters:  {} candidates, {} tested, {} sufficient'
                      .format(len(rasters), counter_tested, counter_sufficient))
        _logger.trace('Rasters used:  {}'.format(', '.join(rasters_used)))
        return raster_data, rasters_used

    def _get_data_from_raster_file(self, filepath, image_dim, point):
        # Calculate bounds by given image with dims == image_dim centered at point == point
        raster = raster_helper.Raster(filepath)
        pt_r, pt_c = raster.RCFromXY(point.bounds[0], point.bounds[1])
        r_start = pt_r + int(image_dim / 2)
        c_start = pt_c - int(image_dim / 2)
        xy_ll = raster.XYFromRC(r_start, c_start)
        xy_ur = raster.XYFromRC(r_start - image_dim + 1, c_start + image_dim - 1)
        bounds = list(xy_ll) + list(xy_ur)
        # Load raster data via image object for heavy lifting
        return geofeature_image.Image(filepath, self.idxs_bands, bounds).raster_data
