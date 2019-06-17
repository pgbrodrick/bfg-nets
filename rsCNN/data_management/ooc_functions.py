

import numpy as np
import os
from typing import List

from rsCNN.utils import logging
import shutil

_MAX_UNIQUE_RESPONSES = 100

_logger = logging.getLogger(__name__)

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


def permute_array(source: np.array, source_filename: str, permutation: np.array) -> np.array:
    perm_memmap_file = os.path.join(
        os.path.dirname(source_filename), str(os.path.basename(source_filename).split('.')[0]) + '_perm.npy')

    shape = source.shape
    type = source.dtype

    dest = np.memmap(perm_memmap_file, dtype=type, mode='w+', shape=shape)

    for i in range(len(permutation)):
        dest[i,...] = source[permutation[i],...]

        if (i % 100 == 0):
            del dest, source
            source = np.memmap(source_filename, dtype=type, shape=shape, mode='r+')
            dest = np.memmap(perm_memmap_file, dtype=type, shape=shape, mode='r+')

    del source, dest

    os.remove(source_filename)
    shutil.move(perm_memmap_file, source_filename)

    source = np.meammap(source_filename, dtype=type, shape=shape, mode='r+')
    return source








