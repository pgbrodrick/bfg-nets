import gdal
import logging
import numpy as np
from numpy import matlib
import os
from typing import List

import numpy as np
import numpy.matlib


_logger = logging.getLogger(__name__)

_MAX_UNIQUE_RESPONSES = 100


def upper_left_pixel(trans, interior_x, interior_y):
    x_ul = max((trans[0] - interior_x)/trans[1], 0)
    y_ul = max((interior_y - trans[3])/trans[5], 0)
    return x_ul, y_ul


def get_overlapping_extent(dataset_list_of_lists: List[List[gdal.Dataset]]):

    # Convert list of lists or list for interior convenience
    dataset_list = [item for sublist in dataset_list_of_lists for item in sublist]

    # Get list of all gdal geotransforms
    trans_list = []
    for _d in range(len(dataset_list)):
        trans_list.append(dataset_list[_d].GetGeoTransform())

    # Find the interior (UL) x,y coordinates in map-space
    interior_x = np.nanmax([x[0] for x in trans_list])
    interior_y = np.nanmin([x[3] for x in trans_list])

    # calculate the UL coordinates in pixel-space
    ul_list = []
    for _d in range(len(dataset_list)):
        ul_list.append(list(upper_left_pixel(trans_list[_d], interior_x, interior_y)))

    # calculate the size of the matched interior extent
    x_len = int(np.floor(np.min([dataset_list[_d].RasterXSize - ul_list[_d][0]
                                 for _d in range(len(dataset_list))])))
    y_len = int(np.floor(np.min([dataset_list[_d].RasterYSize - ul_list[_d][1]
                                 for _d in range(len(dataset_list))])))

    # separate out into list of lists for return
    return_ul_list = []
    idx = 0
    for _l in range(len(dataset_list_of_lists)):
        local_list = []
        for _d in range(len(dataset_list_of_lists[_l])):
            local_list.append(ul_list[idx])
            idx += 1
        local_list = np.array(local_list).astype(int)
        return_ul_list.append(local_list)
    return return_ul_list, x_len, y_len


def get_all_interior_extent_subset_pixel_locations(gdal_datasets: List[List[gdal.Dataset]], window_radius: int, inner_window_radius: int = None, shuffle: bool =True):

        if (inner_window_radius is None):
            inner_window_radius = window_radius

        all_upper_lefts, x_px_size, y_px_size = get_overlapping_extent(gdal_datasets)

        _logger.debug('Calculate pixel-based interior offsets for data acquisition')
        x_sample_locations = [x for x in range(0,
                                          int(x_px_size - 2*window_radius)-1,
                                          int(inner_window_radius*2))]
        y_sample_locations = [y for y in range(0,
                                          int(y_px_size - 2*window_radius)-1,
                                          int(inner_window_radius*2))]

        xy_sample_locations = np.zeros((len(x_sample_locations)*len(y_sample_locations), 2)).astype(int)
        xy_sample_locations[:, 0] = np.matlib.repmat(
            np.array(x_sample_locations).reshape((-1, 1)), 1, len(y_sample_locations)).flatten()
        xy_sample_locations[:, 1] = np.matlib.repmat(
            np.array(y_sample_locations).reshape((1, -1)), len(x_sample_locations), 1).flatten()
        del x_sample_locations, y_sample_locations

        if (shuffle):
            xy_sample_locations = xy_sample_locations[np.random.permutation(xy_sample_locations.shape[0]), :]

        return all_upper_lefts, xy_sample_locations




def read_map_subset(datasets: List, upper_lefts: List[List[int]], window_diameter: int, mask=None, nodata_value=None):
    local_array = np.zeros((window_diameter, window_diameter, np.sum([lset.RasterCount for lset in datasets])))
    if mask is None:
        mask = np.zeros((window_diameter, window_diameter)).astype(bool)
    idx = 0
    for _file in range(len(datasets)):
        file_set = datasets[_file]
        file_upper_left = upper_lefts[_file]
        file_array = np.zeros((window_diameter, window_diameter, file_set.RasterCount))
        for _b in range(file_set.RasterCount):
            file_array[:, :, _b] = file_set.GetRasterBand(
                _b+1).ReadAsArray(int(file_upper_left[0]), int(file_upper_left[1]), window_diameter, window_diameter)

        if (nodata_value is not None):
            file_array[file_array == nodata_value] = np.nan

        file_array[np.isfinite(file_array) is False] = np.nan
        file_array[mask, :] = np.nan

        mask[np.any(np.isnan(file_array), axis=-1)] = True
        if np.all(mask):
            return None, None
        local_array[..., idx:idx+file_array.shape[-1]] = file_array
        idx += file_array.shape[-1]

    return local_array, mask


def one_hot_encode_array(raw_band_types: List[str], array: np.array, memmap_file: str = None):

    cat_band_locations = [idx for idx, val in enumerate(raw_band_types) if val == 'C']
    band_types = raw_band_types.copy()
    for _c in reversed(range(len(cat_band_locations))):

        un_array = array[..., cat_band_locations[_c]]
        un_array = np.unique(un_array[np.isfinite(un_array)])
        assert len(un_array) < _MAX_UNIQUE_RESPONSES,\
            'Too many ({}) unique responses found, suspected incorrect categorical specification'.format(len(un_array))
        _logger.info('Found {} categorical responses'.format(len(un_array)))
        _logger.debug('Cat response: {}'.format(un_array))

        array_shape = list(array.shape)
        array_shape[-1] = len(un_array) + array.shape[-1] - 1

        if (memmap_file is not None):
            cat_memmap_file = os.path.join(
                os.path.dirname(memmap_file), str(os.path.basename(memmap_file).split('.')[0]) + '_cat.npy')
            cat_array = np.memmap(cat_memmap_file,
                                  dtype=np.float32,
                                  mode='w+',
                                  shape=tuple(array_shape))
        else:
            cat_array = np.zeros(tuple(array_shape))

        # One hot-encode
        for _r in range(array_shape[-1]):
            if (_r >= cat_band_locations[_c] and _r < len(un_array)):
                cat_array[..., _r] = np.squeeze(array[..., cat_band_locations[_c]] ==
                                                un_array[_r - cat_band_locations[_c]])
            else:
                if (_r < cat_band_locations[_c]):
                    cat_array[..., _r] = array[..., _r]
                else:
                    cat_array[..., _r] = array[..., _r - len(un_array) + 1]

        # Force file dump, and then reload the encoded responses as the primary response
        del array, cat_array

        if (memmap_file is not None):
            if (os.path.isfile(memmap_file)):
                os.remove(memmap_file)
            memmap_file = cat_memmap_file
            array = np.memmap(memmap_file, dtype=np.float32, mode='r+', shape=tuple(array_shape))

        band_types.pop(cat_band_locations[_c])
        for _r in range(len(un_array)):
            band_types.insert(cat_band_locations[_c], 'B' + str(int(_c)))
    return array, band_types
