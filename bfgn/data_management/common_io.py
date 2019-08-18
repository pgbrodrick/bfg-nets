import os
import copy
import fiona
import gdal
import logging
import numpy as np
from numpy import matlib
import rasterio.features
from typing import List, Tuple

from bfgn.configuration import configs


_logger = logging.getLogger(__name__)


def upper_left_pixel(trans, interior_x, interior_y):
    x_ul = max((interior_x - trans[0])/trans[1], 0)
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

        # special case of no boundary file.  Append a None if no value:
        if (len(local_list) == 0):
            local_list.append(None)

        local_list = np.array(local_list)
        return_ul_list.append(local_list)
    return return_ul_list, x_len, y_len


def get_overlapping_extent_coordinates(dataset_list_of_lists: List[List[gdal.Dataset]]):

    # Convert list of lists or list for interior convenience
    dataset_list = [item for sublist in dataset_list_of_lists for item in sublist]

    # Get list of all gdal geotransforms
    trans_list = []
    for _d in range(len(dataset_list)):
        trans_list.append(dataset_list[_d].GetGeoTransform())

    # Find the interior (UL) x,y coordinates in map-space
    interior_x = np.nanmax([x[0] for x in trans_list])
    interior_y = np.nanmin([x[3] for x in trans_list])

    exterior_x = np.nanmin([trans_list[x][0]+dataset_list[x].RasterXSize*trans_list[x][1]
                            for x in range(len(trans_list))])
    exterior_y = np.nanmax([trans_list[x][3]+dataset_list[x].RasterYSize*trans_list[x][5]
                            for x in range(len(trans_list))])

    return [interior_x, interior_y], [exterior_x, exterior_y]


def get_all_interior_extent_subset_pixel_locations(gdal_datasets: List[List[gdal.Dataset]], window_radius: int, inner_window_radius: int = None, shuffle: bool = True, return_xy_size: bool = False):

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

    if return_xy_size:
        return all_upper_lefts, xy_sample_locations, [x_px_size, y_px_size]
    else:
        return all_upper_lefts, xy_sample_locations


def read_map_subset(datafiles: List[str], upper_lefts: List[List[int]], window_diameter: int, mask=None,
                    nodata_value: float = None, lower_bound: float = None, upper_bound: float = None,
                    reference_geotransform=None):

    raster_count = 0
    datasets = []
    for _f in range(len(datafiles)):
        if (datafiles[_f].split('.')[-1] not in configs.sections.VECTORIZED_FILENAMES):
            datasets.append(gdal.Open(datafiles[_f], gdal.GA_ReadOnly))
            raster_count += datasets[-1].RasterCount
        else:
            datasets.append(None)
            raster_count += 1

    local_array = np.zeros((window_diameter, window_diameter, raster_count))
    if mask is None:
        mask = np.zeros((window_diameter, window_diameter)).astype(bool)

    idx = 0
    for _file in range(len(datasets)):

        if (datasets[_file] is not None):
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

            with np.errstate(invalid='ignore'):
                if (lower_bound is not None):
                    file_array[file_array < lower_bound] = np.nan
                if (upper_bound is not None):
                    file_array[file_array > upper_bound] = np.nan

            mask[np.any(np.isnan(file_array), axis=-1)] = True
        else:
            assert reference_geotransform is not None

            file_array = rasterize_vector(datafiles[_f], reference_geotransform,
                                          (window_diameter, window_diameter, 1))

        if np.all(mask):
            return None, None
        local_array[..., idx:idx+file_array.shape[-1]] = file_array
        idx += file_array.shape[-1]

    return local_array, mask


def get_boundary_sets_from_boundary_files(config: configs.Config) -> List[gdal.Dataset]:
    if not config.raw_files.boundary_files:
        boundary_sets = [None] * len(config.raw_files.feature_files)
    else:
        boundary_sets = [noerror_open(loc_file) if loc_file is not None else None
                         for loc_file in config.raw_files.boundary_files]
    return boundary_sets


def get_site_boundary_set(config: configs.Config, _site) -> gdal.Dataset:
    if not config.raw_files.boundary_files:
        boundary_set = None
    else:
        boundary_set = noerror_open(config.raw_files.boundary_files[_site])

    return boundary_set


def get_site_boundary_vector_file(config: configs.Config, _site) -> str:
    if not config.raw_files.boundary_files:
        boundary_file = None
    else:
        boundary_file = config.raw_files.boundary_files[_site]
        if (boundary_file.split('.')[-1] not in configs.sections.VECTORIZED_FILENAMES):
            boundary_file = None

    return boundary_file


def rasterize_vector(vector_file: str, geotransform: List[float], output_shape: Tuple) -> np.array:
    """ Rasterizes an input vector directly into a numpy array.

    Args:
        vector_file: Input vector file to be rasterized
        geotransform: A gdal style geotransform
        output_shape: The shape of the output file to be generated

    Returns:
        mask: A rasterized 2-d numpy array
    """
    ds = fiona.open(vector_file, 'r')
    trans = copy.deepcopy(geotransform)
    trans = [trans[1], trans[2], trans[0],
             trans[4], trans[5], trans[3]]
    mask = np.zeros(output_shape)
    for n in range(0, len(ds)):
        rasterio.features.rasterize([ds[n]['geometry']], transform=trans, default_value=1, out=mask)
    ds.close()
    return mask


def read_mask_chunk(
        _site: int,
        upper_left: List[int],
        window_diameter: int,
        reference_geotransform: List[float],
        config: configs.Config
) -> np.array:

    mask = None
    boundary_set = get_site_boundary_set(config, _site)
    if (boundary_set is not None):
        mask = boundary_set.ReadAsArray(int(upper_left[0]), int(upper_left[1]), window_diameter, window_diameter)
    else:
        boundary_vector_file = get_site_boundary_vector_file(config, _site)

        if (boundary_vector_file is not None):
            mask = rasterize_vector(boundary_vector_file, reference_geotransform,
                                    (window_diameter, window_diameter))

    if (mask is None):
        mask = np.zeros((window_diameter, window_diameter)).astype(bool)
    else:
        mask = mask == config.raw_files.boundary_bad_value

    return mask


def noerror_open(filename: str, file_handle=gdal.GA_ReadOnly) -> gdal.Dataset:
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    dataset = gdal.Open(filename, file_handle)
    gdal.PopErrorHandler()
    return dataset


def convert_envi_file(original_file: str, destination_basename: str, output_format: str, cleanup: bool = False, creation_options: List = []) -> None:
    """
    Convert an ENVI file to another output format with a gdal_translate call

    Args:
        original_file: Source envi file
        destination_basename: Base of the output file (will get appropriate extension)
        output_format: A viable gdal output data format.
        cleanup: boolean indicating whether or not to cleanup original envi files
        creation_options: GDAL creation options to pass for output file, e.g.: ['TILED=YES', 'COMPRESS=DEFLATE']
    Returns:
         None
    """
    final_outname = destination_basename
    if (output_format == 'GTiff'):
        final_outname += '.tif'
    elif (output_format == 'JPEG'):
        final_outname += '.jpg'
    elif (output_format == 'PNG'):
        final_outname += '.png'
    _logger.debug('Output format {}.  Converting ENVI file to output data with creation options {}'.
                  format(output_format, creation_options))

    options = ''
    for co in creation_options:
        options += ' -co {}'.format(co)
    gdal.Translate(final_outname, gdal.Open(original_file, gdal.GA_ReadOnly), options=options)

    test_outdataset = gdal.Open(final_outname, gdal.GA_ReadOnly)
    if (cleanup):
        if (test_outdataset is not None):
            _logger.debug('Format transform successfull, cleanup ENVI files {}, {}, {}'.
                          format(original_file, original_file + '.hdr', original_file + '.aux.xml'))
            os.remove(original_file)
            os.remove(original_file + '.hdr')
            try:
                os.remove(original_file + '.aux.xml')
            except OSError:
                pass
        else:
            _logger.error('Failed to successfully convert output ENVI file to {}.  ENVI file available at: {}'.
                          format(output_format, original_file))


def read_chunk_by_row(datasets: List[gdal.Dataset], pixel_upper_lefts: List[List[int]], x_size: int, y_size: int,
                      line_offset: int, nodata_value: float = None) -> np.array:
    """
    Read a chunk of multiple datasets line-by-line.

    Args:
        datasets: each feature dataset to read
        pixel_upper_lefts: upper left hand pixel of each dataset
        x_size: size of x data to read
        y_size: size of y data to read
        line_offset: line offset from UL of each set to start reading at
        nodata_value: value to encode to np.nan

    Returns:
        feature_array: feature array
    """

    total_features = np.sum([f.RasterCount for f in datasets])
    output_array = np.zeros((y_size, x_size, total_features))

    feat_ind = 0
    for _feat in range(len(datasets)):
        dat = np.zeros((datasets[_feat].RasterCount, y_size, x_size))
        for _line in range(y_size):
            dat[:, _line:_line+1, :] = datasets[_feat].ReadAsArray(int(pixel_upper_lefts[_feat][0]),
                                                                   int(pixel_upper_lefts[_feat][1] + line_offset + _line), x_size, 1).astype(np.float32)
        # Swap dat from (b,y,x) to (y,x,b)
        dat = np.moveaxis(dat, [0, 1, 2], [2, 0, 1])

        output_array[..., feat_ind:feat_ind+dat.shape[-1]] = dat
        feat_ind += dat.shape[-1]
        if (output_array is not None):
            output_array[output_array == nodata_value] = np.nan

    return output_array
