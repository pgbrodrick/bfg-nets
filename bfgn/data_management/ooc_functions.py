import logging
import numpy as np
import os
from typing import List, Tuple

import shutil

_MAX_UNIQUE_RESPONSES = 100

_logger = logging.getLogger(__name__)


def one_hot_encode_array(
    raw_band_types: List[str],
    array: np.array,
    memmap_file: str = None,
    per_band_encoding: List[np.array] = None,
) -> Tuple[np.array, List[str], List[np.array]]:
    """One hot encode an array of mixed real and categorical variables.

    Args:
        raw_band_types: Band types for given array, either 'R' for real or 'C' for categorical.
        array: array to encode

        memmap_file: file to use to do things out-of-core
        per_band_encoding: if none, this will be calculated and returned.  If not none, these will be used to encode
                           the array

    Returns:
        array: now one-hot-encoded
        band_types: the one-hot-encoded versinon of the band-types
        return_band_encoding: the encoding used on a per-categorical-band basis, if per_band_encoding was None when
                              provided, otherwise None
    """
    cat_band_locations = [idx for idx, val in enumerate(raw_band_types) if val == "C"]
    band_types = raw_band_types.copy()

    if per_band_encoding is None:
        return_band_encoding = []
    else:
        assert len(per_band_encoding) == len(
            cat_band_locations
        ), "Inconsistent lengths of categorical band locations and per_band_encoding provided"
        return_band_encoding = None

    for _c in reversed(range(len(cat_band_locations))):

        if per_band_encoding is None:
            un_array = array[..., cat_band_locations[_c]]
            un_array = np.unique(un_array[np.isfinite(un_array)])
            return_band_encoding.append(un_array)
        else:
            un_array = per_band_encoding[_c]

        assert (
            len(un_array) < _MAX_UNIQUE_RESPONSES
        ), "Too many ({}) unique responses found, suspected incorrect categorical specification".format(
            len(un_array)
        )
        _logger.info("Found {} categorical responses".format(len(un_array)))
        _logger.info("Cat response: {}".format(un_array))

        array_shape = list(array.shape)
        array_shape[-1] = len(un_array) + array.shape[-1] - 1

        if memmap_file is not None:
            cat_memmap_file = os.path.join(
                os.path.dirname(memmap_file),
                str(os.path.splitext(os.path.basename(memmap_file))[0]) + "_cat.npy",
            )
            cat_array = np.memmap(
                cat_memmap_file, dtype=np.float32, mode="w+", shape=tuple(array_shape)
            )
        else:
            cat_array = np.zeros(tuple(array_shape))

        # One hot-encode
        for _r in range(array_shape[-1]):
            if _r >= cat_band_locations[_c] and _r < len(un_array):
                cat_array[..., _r] = np.squeeze(
                    array[..., cat_band_locations[_c]]
                    == un_array[_r - cat_band_locations[_c]]
                )
            else:
                if _r < cat_band_locations[_c]:
                    cat_array[..., _r] = array[..., _r]
                else:
                    cat_array[..., _r] = array[..., _r - len(un_array) + 1]

        # Force file dump, and then reload the encoded responses as the primary response
        del array, cat_array

        if memmap_file is not None:
            if os.path.isfile(memmap_file):
                os.remove(memmap_file)
            memmap_file = cat_memmap_file
            array = np.memmap(
                memmap_file, dtype=np.float32, mode="r+", shape=tuple(array_shape)
            )

        band_types.pop(cat_band_locations[_c])
        for _r in range(len(un_array)):
            band_types.insert(cat_band_locations[_c], "B" + str(int(_c)))

    if per_band_encoding is not None:
        return array
    else:
        return array, band_types, return_band_encoding


def permute_array(
    source: np.array, source_filename: str, permutation: np.array
) -> np.array:
    perm_memmap_file = os.path.join(
        os.path.dirname(source_filename),
        str(os.path.splitext(os.path.basename(source_filename))[0]) + "_perm.npy",
    )

    shape = source.shape
    dtype = source.dtype

    dest = np.memmap(perm_memmap_file, dtype=dtype, mode="w+", shape=shape)

    for i in range(len(permutation)):
        dest[i, ...] = source[permutation[i], ...]

        if i % 100 == 0:
            del dest, source
            source = np.memmap(source_filename, dtype=dtype, shape=shape, mode="r+")
            dest = np.memmap(perm_memmap_file, dtype=dtype, shape=shape, mode="r+")

    del source, dest

    os.remove(source_filename)
    shutil.move(perm_memmap_file, source_filename)

    source = np.memmap(source_filename, dtype=dtype, shape=shape, mode="r+")
    return source
