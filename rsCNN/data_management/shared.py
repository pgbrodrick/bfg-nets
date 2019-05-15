
import numpy as np
import gdal
from typing import List, Tuple
from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)


def read_map_subset(datasets: List, upper_lefts: List[List[int]], window_diameter: int, mask=None, nodata_value=None):
    local_array = np.zeros((window_diameter, window_diameter, np.sum([lset.RasterCount for lset in datasets])))
    if mask is None:
        mask = np.zeros((window_diameter, window_diameter))
    idx = 0
    for _file in range(len(datasets)):
        file_set = datasets[_file]
        file_upper_left = upper_lefts[_file]
        file_array = np.zeros((window_diameter, window_diameter, file_set.RasterCount))
        for _b in range(file_set.RasterCount):
            file_array[:, :, _b] = file_set.GetRasterBand(
                _b+1).ReadAsArray(file_upper_left[0], file_upper_left[1], window_diameter, window_diameter)

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
        assert len(un_array) < MAX_UNIQUE_RESPONSES,\
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
