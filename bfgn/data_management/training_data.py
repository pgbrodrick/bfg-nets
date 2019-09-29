import copy
import fiona
import gdal
import logging
import numpy as np
import ogr
import osr
import os
from tqdm import tqdm
from typing import List, Tuple

from bfgn.configuration import configs, sections
from bfgn.data_management import common_io, ooc_functions, data_core

_logger = logging.getLogger(__name__)


def build_training_data_ordered(
    config: configs.Config,
    feature_raw_band_types: List[str],
    response_raw_band_types: List[str],
) -> Tuple[
    List[np.array],
    List[np.array],
    List[np.array],
    List[str],
    List[str],
    List[np.array],
    List[np.array],
]:

    # TODO:  check default, and set it to a standard value
    if config.data_build.random_seed:
        _logger.debug("Setting random seed to {}".format(config.data_build.random_seed))
        np.random.seed(config.data_build.random_seed)

    # TODO:  move to config checks
    if isinstance(config.data_build.max_samples, list):
        if len(config.data_build.max_samples) != len(config.raw_files.feature_files):
            raise Exception(
                "max_samples must equal feature_files length, or be an integer."
            )

    n_features = int(np.sum([len(feat_type) for feat_type in feature_raw_band_types]))
    n_responses = int(np.sum([len(resp_type) for resp_type in response_raw_band_types]))

    # TODO:  move to checks
    feature_memmap_size_gb = (
        n_features
        * 4
        * config.data_build.max_samples
        * (config.data_build.window_radius * 2) ** 2
        / 1024.0 ** 3
    )
    assert (
        feature_memmap_size_gb < config.data_build.max_built_data_gb
    ), "Expected feature memmap size: {} Gb, limit: {}".format(
        feature_memmap_size_gb, config.data_build.max_built_data_gb
    )

    response_memmap_size_gb = (
        n_responses
        * 4
        * config.data_build.max_samples
        * (config.data_build.window_radius * 2) ** 2
        / 1024.0 ** 3
    )
    assert (
        response_memmap_size_gb < config.data_build.max_built_data_gb
    ), "Expected feature memmap size: {} Gb, limit: {}".format(
        response_memmap_size_gb, config.data_build.max_built_data_gb
    )

    features, responses = _open_temporary_features_responses_data_files(
        config, n_features, n_responses
    )
    _log_munged_data_information(features, responses)

    _logger.debug("Pre-compute all subset locations")
    all_site_upper_lefts = []
    all_site_xy_locations = []
    all_site_xy_sizes = []
    reference_subset_geotransforms = []
    for _site in range(0, len(config.raw_files.feature_files)):
        feature_sets = [
            gdal.Open(loc_file, gdal.GA_ReadOnly)
            for loc_file in config.raw_files.feature_files[_site]
        ]
        response_sets = [
            common_io.noerror_open(loc_file)
            for loc_file in config.raw_files.response_files[_site]
        ]
        boundary_set = common_io.get_site_boundary_set(config, _site)

        all_set_upper_lefts, xy_sample_locations, xy_max_size = common_io.get_all_interior_extent_subset_pixel_locations(
            gdal_datasets=[
                feature_sets,
                [rs for rs in response_sets if rs is not None],
                [bs for bs in [boundary_set] if bs is not None],
            ],
            window_radius=config.data_build.window_radius,
            inner_window_radius=config.data_build.loss_window_radius,
            shuffle=True,
            return_xy_size=True,
        )

        all_site_upper_lefts.append(all_set_upper_lefts)
        all_site_xy_locations.append(xy_sample_locations)
        all_site_xy_sizes.append(xy_max_size)

        ref_trans = feature_sets[0].GetGeoTransform()
        ref_subset_geotransform = [
            ref_trans[0] + all_set_upper_lefts[0][0][0] * ref_trans[1],
            ref_trans[1],
            0,
            ref_trans[3] + all_set_upper_lefts[0][0][1] * ref_trans[5],
            0,
            ref_trans[5],
        ]
        reference_subset_geotransforms.append(ref_subset_geotransform)

    if config.data_build.sparse_read is True:
        _logger.info(
            "Sparse read is on, so use a pass the response file to thin down the training points to read"
        )
        for _site in range(0, len(config.raw_files.feature_files)):
            response_sets = [
                common_io.noerror_open(loc_file)
                for loc_file in config.raw_files.response_files[_site]
            ]
            boundary_set = common_io.get_site_boundary_set(config, _site)

            valid_locations = np.ones(all_site_xy_locations[_site].shape[0]).astype(
                bool
            )
            for _row in tqdm(
                range(0, all_site_xy_sizes[_site][1]),
                ncols=80,
                desc="Sparse-data filter",
            ):

                subset = np.squeeze(
                    np.where(all_site_xy_locations[_site][:, 1] == _row)
                )
                if len(subset) > 0:
                    if boundary_set is not None:
                        bound_dat = common_io.read_chunk_by_row(
                            boundary_set,
                            all_site_upper_lefts[_site][-1],
                            all_site_xy_sizes[_site][0],
                            config.data_build.loss_window_radius * 2,
                            _row,
                            nodata_value=config.raw_files.boundary_bad_value,
                        )

                        for _x_loc in range(len(subset)):
                            x_loc = all_site_xy_locations[_site][subset[_x_loc], 0]
                            boundary_fraction = np.sum(
                                np.isnan(
                                    bound_dat[
                                        :,
                                        x_loc : x_loc
                                        + config.data_build.loss_window_radius * 2,
                                    ]
                                )
                            ) / float(config.data_build.loss_window_radius ** 2)
                            _logger.debug(
                                "Sparsity check boundary fraction: {}".format(
                                    boundary_fraction
                                )
                            )
                            if (
                                boundary_fraction
                                > config.data_build.response_nodata_maximum_fraction
                            ):
                                valid_locations[subset[_x_loc]] = False

                    if np.any(valid_locations[subset]):

                        col_dat = common_io.read_chunk_by_row(
                            response_sets,
                            all_site_upper_lefts[_site][
                                len(config.raw_files.feature_files[_site])
                            ],
                            all_site_xy_sizes[_site][0],
                            config.data_build.loss_window_radius * 2,
                            _row,
                            nodata_value=config.raw_files.response_nodata_value,
                        )

                        for _x_loc in range(len(subset)):
                            x_loc = all_site_xy_locations[_site][subset[_x_loc], 0]
                            response_fractions = np.sum(
                                np.isnan(
                                    col_dat[
                                        :,
                                        x_loc : x_loc
                                        + config.data_build.loss_window_radius * 2,
                                    ]
                                ),
                                axis=(0, 1),
                            ) / float(config.data_build.loss_window_radius ** 2)
                            _logger.debug(
                                "Sparsity check response fraction: {}".format(
                                    response_fractions
                                )
                            )
                            if np.any(
                                response_fractions
                                > config.data_build.response_nodata_maximum_fraction
                            ):
                                valid_locations[subset[_x_loc]] = False
            _logger.info(
                "Filtering {} out of {} points from sprasity check.".format(
                    len(valid_locations) - np.sum(valid_locations), len(valid_locations)
                )
            )
            all_site_xy_locations[_site] = all_site_xy_locations[_site][
                valid_locations, :
            ]

    num_reads_per_site = int(
        np.floor(config.data_build.max_samples / len(config.raw_files.feature_files))
    )

    _logger.debug(
        "Step through sites in order and grab {} sample from each until max samples".format(
            num_reads_per_site
        )
    )
    progress_bar = tqdm(
        desc="Reading data chunks",
        total=config.data_build.max_samples,
        ncols=80,
        mininterval=1,
    )

    _num_samples_total = 0

    for idx_site in range(len(config.raw_files.feature_files)):
        _logger.debug("Reading loop: Site {}".format(idx_site))
        _num_samples_site = 0

        for idx_xy, xy_location in enumerate(all_site_xy_locations[idx_site]):
            _logger.debug("Site index: {}".format(idx_xy))

            success = read_segmentation_chunk(
                idx_site,
                all_site_upper_lefts[idx_site],
                xy_location.copy(),
                config,
                reference_subset_geotransforms[idx_site],
                _num_samples_total,
            )

            progress_bar.refresh()
            if success is True:
                progress_bar.update(1)
                _num_samples_site += 1
                _num_samples_total += 1

            is_site_sampling_complete = _num_samples_site == num_reads_per_site
            is_total_sampling_complete = (
                _num_samples_total == config.data_build.max_samples
            )
            if is_site_sampling_complete or is_total_sampling_complete:
                break

        if is_total_sampling_complete:
            break

        _logger.debug("All locations in site {} have been checked.".format(idx_site))

    progress_bar.close()
    del all_site_upper_lefts
    del all_site_xy_locations
    del reference_subset_geotransforms

    features, responses = _open_temporary_features_responses_data_files(
        config, n_features, n_responses, read_type="r+"
    )
    features = _resize_munged_features(features, _num_samples_total, config)
    responses = _resize_munged_responses(responses, _num_samples_total, config)
    assert features.shape[0] > 0, "Failed to collect a single sample"
    _log_munged_data_information(features, responses)

    _logger.debug("Shuffle data to avoid fold assignment biases")
    perm = np.random.permutation(features.shape[0])
    features = ooc_functions.permute_array(
        features, data_core.get_temporary_features_filepath(config), perm
    )
    responses = ooc_functions.permute_array(
        responses, data_core.get_temporary_responses_filepath(config), perm
    )
    del perm

    _logger.debug("Create uniform weights")
    shape = tuple(list(features.shape)[:-1] + [1])
    weights = np.memmap(
        data_core.get_temporary_weights_filepath(config),
        dtype=np.float32,
        mode="w+",
        shape=shape,
    )
    weights[:, :, :, :] = 1
    _logger.debug("Remove weights for missing responses")
    weights[np.isnan(responses[..., 0])] = 0

    _logger.debug("Remove weights outside loss window")
    if config.data_build.loss_window_radius != config.data_build.window_radius:
        buf = config.data_build.window_radius - config.data_build.loss_window_radius
        weights[:, :buf, :, -1] = 0
        weights[:, -buf:, :, -1] = 0
        weights[:, :, :buf, -1] = 0
        weights[:, :, -buf:, -1] = 0
    _log_munged_data_information(features, responses, weights)

    _logger.debug("One-hot encode features")
    features, feature_band_types, feature_per_band_encoded_values = ooc_functions.one_hot_encode_array(
        feature_raw_band_types,
        features,
        data_core.get_temporary_features_filepath(config),
    )
    _logger.debug("One-hot encode responses")
    responses, response_band_types, response_per_band_encoded_values = ooc_functions.one_hot_encode_array(
        response_raw_band_types,
        responses,
        data_core.get_temporary_responses_filepath(config),
    )
    _log_munged_data_information(features, responses, weights)

    _save_built_data_files(features, responses, weights, config)
    del features, responses, weights

    if "C" in response_raw_band_types:
        assert (
            np.sum(np.array(response_raw_band_types) == "C") == 1
        ), "Weighting is currently only enabled for one categorical response variable."
        features, responses, weights = load_built_data_files(config, writeable=True)
        weights = calculate_categorical_weights(responses, weights, config)
        _logger.debug("Delete in order to flush output")
        del features, responses, weights

    _remove_temporary_data_files(config)

    _logger.debug("Store data build config sections")
    _save_built_data_config_sections_to_verify_successful(config)
    features, responses, weights = load_built_data_files(config, writeable=False)
    return (
        features,
        responses,
        weights,
        feature_band_types,
        response_band_types,
        feature_per_band_encoded_values,
        response_per_band_encoded_values,
    )


def build_training_data_from_response_points(
    config: configs.Config,
    feature_raw_band_types: List[str],
    response_raw_band_types: List[str],
) -> Tuple[
    List[np.array],
    List[np.array],
    List[np.array],
    List[str],
    List[str],
    List[np.array],
    List[np.array],
]:

    _logger.info("Build training data from response points")
    if config.data_build.random_seed is not None:
        np.random.seed(config.data_build.random_seed)

    num_features = np.sum([len(feat_type) for feat_type in feature_raw_band_types])

    feature_memmap_size_gb = (
        num_features
        * 4
        * config.data_build.max_samples
        * (config.data_build.window_radius * 2) ** 2
        / 1024.0 ** 3
    )
    assert (
        feature_memmap_size_gb < config.data_build.max_built_data_gb
    ), "Expected feature memmap size: {} Gb, limit: {}".format(
        feature_memmap_size_gb, config.data_build.max_built_data_gb
    )

    xy_sample_points_per_site = []
    responses_per_site = []
    reference_subset_geotransforms = []
    for _site in range(0, len(config.raw_files.feature_files)):

        _logger.debug("Run through first response")
        if (
            config.raw_files.response_files[_site][0].split(".")[-1]
            in sections.VECTORIZED_FILENAMES
        ):
            assert (
                len(config.raw_files.response_files[_site]) == 1
            ), "Only 1 vector file per site supported for CNN"

            feature_sets = [
                gdal.Open(loc_file, gdal.GA_ReadOnly)
                for loc_file in config.raw_files.feature_files[_site]
            ]
            boundary_set = common_io.get_site_boundary_set(config, _site)

            lower_coord, upper_coord = common_io.get_overlapping_extent_coordinates(
                [feature_sets, [], [bs for bs in [boundary_set] if bs is not None]]
            )

            resolution = np.squeeze(np.array(feature_sets[0].GetGeoTransform()))[
                np.array([1, 5])
            ]

            vector = fiona.open(config.raw_files.response_files[_site][0])
            x_sample_points = []
            y_sample_points = []
            band_responses = []
            for _sample in tqdm(range(len(vector)), ncols=80, desc="Reading responses"):
                if vector[_sample]["geometry"]["type"] == "Point":
                    x_sample_points.append(
                        vector[_sample]["geometry"]["coordinates"][0]
                    )
                    y_sample_points.append(
                        vector[_sample]["geometry"]["coordinates"][1]
                    )
                    band_responses.append(
                        vector[_sample]["properties"][
                            config.raw_files.response_vector_property_name
                        ]
                    )

                    if (
                        x_sample_points[-1]
                        < lower_coord[0]
                        + resolution[0] * config.data_build.window_radius
                        or x_sample_points[-1]
                        > upper_coord[0]
                        - resolution[0] * config.data_build.window_radius
                        or y_sample_points[-1]
                        < upper_coord[1]
                        + resolution[1] * config.data_build.window_radius
                        or y_sample_points[-1]
                        > lower_coord[1]
                        - resolution[1] * config.data_build.window_radius
                    ):

                        x_sample_points.pop(-1)
                        y_sample_points.pop(-1)
                        band_responses.pop(-1)
                else:
                    _logger.info(
                        "WARNING: Non point gemoetry ignored in file {}".format(
                            config.raw_files.response_files[_site][0]
                        )
                    )
            vector.close()

            xy_sample_points = np.vstack(
                [np.array(x_sample_points), np.array(y_sample_points)]
            ).T
            responses_per_file = np.array(np.array(band_responses).reshape(-1, 1))

            reference_subset_geotransforms.append(
                list(feature_sets[0].GetGeoTransform())
            )

        else:
            # TODO: Needs completion
            assert False, "Mode not yet supported"

        xy_sample_points_per_site.append(xy_sample_points)
        responses_per_site.append(responses_per_file)

    total_samples = sum(
        [site_responses.shape[0] for site_responses in responses_per_site]
    )
    _logger.debug(
        "Found {} total samples across {} sites".format(
            total_samples, len(responses_per_site)
        )
    )

    assert total_samples > 0, "need at least 1 valid sample..."

    if total_samples > config.data_build.max_samples:
        _logger.debug(
            "Discard samples because the number of valid samples ({}) exceeds the max samples requested ({})".format(
                total_samples, config.data_build.max_samples
            )
        )

        prop_samples_kept_per_site = config.data_build.max_samples / total_samples
        for _site in range(len(responses_per_site)):
            num_samples = len(responses_per_site[_site])
            num_samples_kept = int(prop_samples_kept_per_site * num_samples)

            idxs_kept = np.random.permutation(num_samples)[:num_samples_kept]
            responses_per_site[_site] = responses_per_site[_site][idxs_kept]

            xy_sample_points_per_site[_site] = xy_sample_points_per_site[_site][
                idxs_kept, :
            ]
            _logger.debug(
                "Site {} had {} valid samples, kept {} samples".format(
                    _site, num_samples, num_samples_kept
                )
            )

        total_samples = sum(
            [site_responses.shape[0] for site_responses in responses_per_site]
        )
        _logger.debug(
            "Kept {} total samples across {} sites after discarding".format(
                total_samples, len(responses_per_site)
            )
        )

    unique_responses = []
    for _site in range(len(responses_per_site)):
        un_site = np.unique(responses_per_site[_site]).tolist()
        unique_responses.extend(un_site)
    unique_responses = np.unique(unique_responses)

    for _response in range(len(unique_responses)):
        _logger.info(
            "Response: {} encoded as: {}".format(unique_responses[_response], _response)
        )

    for _site in range(len(responses_per_site)):
        revised_responses = np.zeros(responses_per_site[_site].shape)
        for _response in range(len(unique_responses)):
            revised_responses[
                responses_per_site[_site] == unique_responses[_response]
            ] = _response
        responses_per_site[_site] = revised_responses.copy()

    features = np.memmap(
        data_core.get_temporary_features_filepath(config),
        dtype=np.float32,
        mode="w+",
        shape=(
            total_samples,
            2 * config.data_build.window_radius,
            2 * config.data_build.window_radius,
            num_features,
        ),
    )

    sample_index = 0
    for _site in range(0, len(config.raw_files.feature_files)):
        _logger.debug("Open feature and response datasets for site {}".format(_site))
        feature_sets = [
            gdal.Open(loc_file, gdal.GA_ReadOnly)
            for loc_file in config.raw_files.feature_files[_site]
        ]
        boundary_set = common_io.get_site_boundary_set(config, _site)

        _logger.debug("Calculate interior rectangle location and extent")
        lower_coord, upper_coord = common_io.get_overlapping_extent_coordinates(
            [feature_sets, [], [bs for bs in [boundary_set] if bs is not None]]
        )

        resolution = np.squeeze(np.array(feature_sets[0].GetGeoTransform()))[
            np.array([1, 5])
        ]

        # xy_sample_locations is current the response centers, but we need to use the pixel ULs.  So subtract
        # out the corresponding feature radius
        # xy_sample_locations = (xy_sample_points_per_site[_site] - np.array(lower_coord))*resolution - \
        # config.data_build.window_radius
        xy_sample_locations = xy_sample_points_per_site[_site]

        ref_trans = feature_sets[0].GetGeoTransform()
        subset_geotransform = None
        if config.raw_files.boundary_files is not None:
            if config.raw_files.boundary_files[_site] is not None and _is_vector_file(
                config.raw_files.boundary_files[_site]
            ):
                subset_geotransform = [
                    ref_trans[0],
                    ref_trans[1],
                    0,
                    ref_trans[3],
                    0,
                    ref_trans[5],
                ]

        good_response_data = np.zeros(responses_per_site[_site].shape[0]).astype(bool)
        # Now read in features
        for _cr in tqdm(
            range(len(xy_sample_locations)), ncols=80, desc="Reading features"
        ):

            local_xy_px_locations = []
            if boundary_set is not None:
                gt = boundary_set.GetGeoTransform()
                local_xy_px_locations.append(
                    [
                        int((xy_sample_locations[_cr, 0] - gt[0]) / gt[1]),
                        int((xy_sample_locations[_cr, 1] - gt[3]) / gt[5]),
                    ]
                )
            else:
                local_xy_px_locations.append(
                    [0, 0]
                )  # doesn't matter the value, won't get used

            for _file in range(len(feature_sets)):
                gt = feature_sets[_file].GetGeoTransform()
                local_xy_px_locations.append(
                    [
                        int((xy_sample_locations[_cr, 0] - gt[0]) / gt[1]),
                        int((xy_sample_locations[_cr, 1] - gt[3]) / gt[5]),
                    ]
                )

            local_feature = read_labeling_chunk(
                _site,
                local_xy_px_locations,
                config,
                reference_subset_geotransforms[_site],
            )

            # Make sure that the feature space also has data - the fact that the response space had valid data is no
            # guarantee that the feature space does.
            if local_feature is not None:
                _logger.debug(
                    "Save sample data; {} samples saved total".format(sample_index + 1)
                )
                features[sample_index, ...] = local_feature.copy()
                good_response_data[_cr] = True
                sample_index += 1
        responses_per_site[_site] = responses_per_site[_site][good_response_data]
        _logger.debug(
            "{} samples saved for site {}".format(
                responses_per_site[_site].shape[0], _site
            )
        )

    assert (
        sample_index > 0
    ), "Insufficient feature data corresponding to response data.  Consider increasing maximum feature nodata size"

    # transform responses
    responses_raw = np.vstack(responses_per_site).astype(np.float32)
    del responses_per_site
    responses = np.memmap(
        data_core.get_temporary_responses_filepath(config),
        dtype=np.float32,
        mode="w+",
        shape=(sample_index, 1),
    )
    responses[:] = responses_raw[:]
    del responses_raw

    _log_munged_data_information(features, responses)

    features = _resize_munged_features(features, sample_index, config)
    _log_munged_data_information(features, responses)

    _logger.debug("Shuffle data to avoid fold assignment biases")
    perm = np.random.permutation(features.shape[0])
    features = ooc_functions.permute_array(
        features, data_core.get_temporary_features_filepath(config), perm
    )
    responses = ooc_functions.permute_array(
        responses, data_core.get_temporary_responses_filepath(config), perm
    )
    del perm

    weights = np.ones((responses.shape[0], 1))
    _log_munged_data_information(features, responses, weights)

    # one hot encode
    features, feature_band_types, feature_per_band_encoded_values = ooc_functions.one_hot_encode_array(
        feature_raw_band_types,
        features,
        data_core.get_temporary_features_filepath(config),
    )
    responses, response_band_types, response_per_band_encoded_values = ooc_functions.one_hot_encode_array(
        response_raw_band_types,
        responses,
        data_core.get_temporary_responses_filepath(config),
    )
    _log_munged_data_information(features, responses, weights)

    # This change happens to work in this instance, but will not work with all sampling strategies.  I will leave for
    # now and refactor down the line as necessary.  Generally speaking, fold assignments are specific to the style of data read
    _save_built_data_files(features, responses, weights, config)
    del features, responses, weights

    if "C" in response_raw_band_types:
        assert (
            np.sum(np.array(response_raw_band_types) == "C") == 1
        ), "Weighting is currently only enabled for one categorical response variable."
        features, responses, weights = load_built_data_files(config, writeable=True)
        weights = calculate_categorical_weights(responses, weights, config)
        del features, responses, weights

    _remove_temporary_data_files(config)

    _save_built_data_config_sections_to_verify_successful(config)
    features, responses, weights = load_built_data_files(config, writeable=False)
    return (
        features,
        responses,
        weights,
        feature_band_types,
        response_band_types,
        feature_per_band_encoded_values,
        response_per_band_encoded_values,
    )


def get_proj(fname: str) -> str:
    """
    Get the projection of a raster/vector dataset.

    :param str fname: Name of input file

    :return The projection of the input fname
    """
    ds = common_io.noerror_open(fname)
    if ds is not None:
        proj = ds.GetProjection()
        if proj is not None:
            srs = osr.SpatialReference(wkt=str(proj))
            if srs.IsProjected is False:
                exception_str = "File {} is not projected.  Correct or set ignore_projections flag.".format(
                    fname
                )
                raise Exception(exception_str)
            return srs.GetAttrValue("projcs")

    if os.path.basename(fname).split(".")[-1] == "shp":
        vset = ogr.GetDriverByName("ESRI Shapefile").Open(fname, gdal.GA_ReadOnly)
    elif os.path.basename(fname).split(".")[-1] == "kml":
        vset = ogr.GetDriverByName("KML").Open(fname, gdal.GA_ReadOnly)
    else:
        exception_str = "Cannot find projection from file {}".format(fname)
        raise Exception(exception_str)

    if vset is None:
        exception_str = "Cannot find projection from file {}".format(fname)
        raise Exception(exception_str)
    else:
        proj = vset.GetLayer().GetSpatialRef()
        if proj is None:
            exception_str = "Cannot find projection from file {}".format(fname)
            raise Exception(exception_str)
        else:
            srs = osr.SpatialReference(wkt=str(proj))
            if srs.IsProjected is False:
                exception_str = "File {} is not projected.  Correct or set ignore_projections flag.".format(
                    fname
                )
                raise Exception(exception_str)
            return srs.GetAttrValue("projcs")


def check_projections(
    f_files: List[List[str]], r_files: List[List[str]], b_files: List[str] = None
) -> List[str]:

    # f = feature, r = response, b = boundary
    errors = []

    if b_files is None:
        b_files = []

    site_f_proj = []
    site_r_proj = []
    site_b_proj = []
    for _site in range(len(f_files)):

        f_proj = []
        r_proj = []
        for _file in range(len(f_files[_site])):
            f_proj.append(get_proj(f_files[_site][_file]))

        for _file in range(len(r_files[_site])):
            r_proj.append(get_proj(r_files[_site][_file]))

        b_proj = None
        if len(b_files) > 0:
            b_proj = get_proj(b_files[_site])

        un_f_proj = np.unique(f_proj)
        if len(un_f_proj) > 1:
            errors.append(
                "Feature projection mismatch at site {}, projections: {}".format(
                    _site, un_f_proj
                )
            )

        un_r_proj = np.unique(r_proj)
        if len(un_r_proj) > 1:
            errors.append(
                "Response projection mismatch at site {}, projections: {}".format(
                    _site, un_r_proj
                )
            )

        if un_f_proj[0] != un_r_proj[0]:
            errors.append(
                "Feature/Response projection mismatch at site {}\nFeature proj: {}\nResponse proj: {}".format(
                    _site, un_f_proj[0], un_r_proj[0]
                )
            )

        if b_proj is not None:
            if un_f_proj[0] != b_proj:
                errors.append(
                    "Feature/Boundary projection mismatch at site {}\nFeature proj: {}\nBoundary proj: {}".format(
                        _site, un_f_proj[0], b_proj
                    )
                )
            if un_r_proj[0] != b_proj:
                errors.append(
                    "Response/Boundary projection mismatch at site {}\nFeature proj: {}\nBoundary proj: {}".format(
                        _site, un_r_proj[0], b_proj
                    )
                )

        site_f_proj.append(un_f_proj[0])
        site_r_proj.append(un_r_proj[0])
        if b_proj is not None:
            site_b_proj.append(b_proj)

    if len(np.unique(site_f_proj)) > 1:
        errors.append(
            "Warning, different projections at different features sites: {}".format(
                site_f_proj
            )
        )
    if len(np.unique(site_r_proj)) > 1:
        errors.append(
            "Warning, different projections at different features sites: {}".format(
                site_r_proj
            )
        )
    if len(np.unique(site_b_proj)) > 1:
        errors.append(
            "Warning, different projections at different features sites: {}".format(
                site_b_proj
            )
        )
    return errors


def check_resolutions(
    f_files: List[List[str]], r_files: List[List[str]], b_files: List[str] = None
) -> List[str]:
    # f = feature, r = response, b = boundary
    errors = []

    if b_files is None:
        b_files = []

    site_f_res = []
    site_r_res = []
    site_b_res = []
    for _site in range(len(f_files)):

        f_res = []
        r_res = []
        for _file in range(len(f_files[_site])):
            f_res.append(
                np.array(
                    gdal.Open(f_files[_site][_file], gdal.GA_ReadOnly).GetGeoTransform()
                )[[1, 5]]
            )

        for _file in range(len(r_files[_site])):
            if not _is_vector_file(r_files[_site][_file]):
                r_res.append(
                    np.array(
                        gdal.Open(
                            r_files[_site][_file], gdal.GA_ReadOnly
                        ).GetGeoTransform()
                    )[[1, 5]]
                )

        b_res = None
        if len(b_files) > 0:
            if not _is_vector_file(b_files[_site]):
                b_res = np.array(
                    gdal.Open(b_files[_site], gdal.GA_ReadOnly).GetGeoTransform()
                )[[1, 5]]

        un_f_res = []
        if len(f_res) > 0:
            f_res = np.vstack(f_res)
            un_f_res = [np.unique(f_res[:, 0]), np.unique(f_res[:, 1])]
            if len(un_f_res[0]) > 1 or len(un_f_res[1]) > 1:
                errors.append(
                    "Feature resolution mismatch at site {}, resolutions: {}".format(
                        _site, un_f_res
                    )
                )

        un_r_res = []
        if len(r_res) > 0:
            r_res = np.vstack(r_res)
            un_r_res = [np.unique(r_res[:, 0]), np.unique(r_res[:, 1])]
            if len(un_r_res[0]) > 1 or len(un_r_res[1]) > 1:
                errors.append(
                    "Response resolution mismatch at site {}, resolutions: {}".format(
                        _site, un_r_res
                    )
                )

        if b_res is not None:
            if un_f_res[0][0] != b_res[0] or un_f_res[1][0] != b_res[1]:
                errors.append(
                    "Feature/Boundary resolution mismatch at site {}".format(_site)
                )
            if len(un_r_res) > 0:
                if un_r_res[0][0] != b_res[0] or un_r_res[1][0] != b_res[1]:
                    errors.append(
                        "Response/Boundary resolution mismatch at site {}".format(_site)
                    )

        site_f_res.append(un_f_res[0])
        if len(un_r_res) > 0:
            site_r_res.append(un_r_res[0])
        if b_res is not None:
            site_b_res.append(b_res)

    if len(np.unique(site_f_res)) > 1:
        _logger.info(
            "Warning, different resolutions at different features sites: {}".format(
                site_f_res
            )
        )
    if len(np.unique(site_r_res)) > 1:
        _logger.info(
            "Warning, different resolutions at different features sites: {}".format(
                site_r_res
            )
        )
    if len(np.unique(site_b_res)) > 1:
        _logger.info(
            "Warning, different resolutions at different features sites: {}".format(
                site_b_res
            )
        )
    return errors


# Calculates categorical weights for a single response
def calculate_categorical_weights(
    responses: List[np.array],
    weights: List[np.array],
    config: configs.Config,
    batch_size: int = 100,
) -> List[np.array]:

    # find upper and lower boud
    lb = config.data_build.window_radius - config.data_build.loss_window_radius
    ub = -lb

    # get response/total counts (batch-wise)
    response_counts = np.zeros(responses[0].shape[-1])
    total_valid_count = 0
    for idx_array, response_array in enumerate(responses):
        if idx_array in (
            config.data_build.validation_fold,
            config.data_build.test_fold,
        ):
            continue
        for ind in range(0, response_array.shape[0], batch_size):
            if lb == 0 or len(response_array.shape) < 3:
                lr = response_array[ind : ind + batch_size, ...]
            else:
                lr = response_array[ind : ind + batch_size, lb:ub, lb:ub, ...]
            lr[lr == config.raw_files.response_nodata_value] = np.nan
            total_valid_count += np.sum(np.isfinite(lr))
            for _r in range(0, len(response_counts)):
                response_counts[_r] += np.nansum(lr[..., _r] == 1)

    # assign_weights
    for _array in range(len(responses)):
        for ind in range(0, responses[_array].shape[0], batch_size):

            lr = (responses[_array])[ind : ind + batch_size, ...]
            lrs = list(lr.shape)
            lrs.pop(-1)
            lw = np.zeros((lrs))
            for _r in range(0, len(response_counts)):
                lw[lr[..., _r] == 1] = total_valid_count / response_counts[_r]

            if lb != 0 and len(lw.shape) == 3:
                lw[:, :lb, :] = 0
                lw[:, ub:, :] = 0
                lw[:, :, :lb] = 0
                lw[:, :, ub:] = 0

            lws = list(lw.shape)
            lws.extend([1])
            lw = lw.reshape(lws)
            weights[_array][ind : ind + batch_size, ...] = lw

    return weights


def read_labeling_chunk(
    _site: int,
    offset_from_ul: List[List[int]],
    config: configs.Config,
    reference_geotransform: List[float],
) -> np.array:

    window_diameter = config.data_build.window_radius * 2

    geotransform = copy.deepcopy(reference_geotransform)
    geotransform[0] += offset_from_ul[0][0] * geotransform[1]
    geotransform[3] += offset_from_ul[0][1] * geotransform[5]

    # for _f in range(len(offset_from_ul)):
    #    print(f_ul[_f])
    #    if (np.any(np.array(offset_from_ul[_f]) < config.data_build.window_radius)):
    #        _logger.debug('Feature read OOB')
    #        return None
    #    if (f_ul[_f][0] > f_sets[_f].RasterXSize - config.data_build.window_radius):
    #        _logger.debug('Feature read OOB')
    #        return None
    #    if (f_ul[_f][1] > f_sets[_f].RasterYSize - config.data_build.window_radius):
    #        _logger.debug('Feature read OOB')
    #        return None

    mask = common_io.read_mask_chunk(
        _site, offset_from_ul.pop(0), window_diameter, geotransform, config
    )

    if not _check_mask_data_sufficient(
        mask, config.data_build.feature_nodata_maximum_fraction
    ):
        return False

    local_feature, mask = common_io.read_map_subset(
        config.raw_files.feature_files[_site],
        offset_from_ul,
        window_diameter,
        mask,
        config.raw_files.feature_nodata_value,
    )

    if not _check_mask_data_sufficient(
        mask, config.data_build.feature_nodata_maximum_fraction
    ):
        _logger.debug("Insufficient feature data")
        return None

    # Final check (propogate mask forward), and return
    local_feature[mask, :] = np.nan

    return local_feature


def read_segmentation_chunk(
    _site: int,
    all_file_upper_lefts: List[List[int]],
    offset_from_ul: List[int],
    config: configs.Config,
    reference_geotransform: List[float],
    sample_index: int,
) -> bool:

    window_diameter = config.data_build.window_radius * 2

    [f_ul, r_ul, [b_ul]] = copy.deepcopy(all_file_upper_lefts)

    geotransform = copy.deepcopy(reference_geotransform)
    geotransform[0] += offset_from_ul[0] * geotransform[1]
    geotransform[3] += offset_from_ul[1] * geotransform[5]

    f_ul += offset_from_ul
    for _r in range(len(r_ul)):
        if r_ul[_r] is not None:
            r_ul[_r] += offset_from_ul

    if b_ul is not None:
        b_ul += offset_from_ul

    mask = common_io.read_mask_chunk(_site, b_ul, window_diameter, geotransform, config)

    if not _check_mask_data_sufficient(
        mask, config.data_build.feature_nodata_maximum_fraction
    ):
        return False

    response_mask = mask.copy()
    local_response, response_mask = common_io.read_map_subset(
        config.raw_files.response_files[_site],
        r_ul,
        window_diameter,
        response_mask,
        config.raw_files.response_nodata_value,
        lower_bound=config.data_build.response_min_value,
        upper_bound=config.data_build.response_max_value,
        reference_geotransform=geotransform,
    )

    if not _check_mask_data_sufficient(
        response_mask, config.data_build.response_nodata_maximum_fraction
    ):
        return False

    feature_mask = mask.copy()
    local_feature, feature_mask = common_io.read_map_subset(
        config.raw_files.feature_files[_site],
        f_ul,
        window_diameter,
        feature_mask,
        config.raw_files.feature_nodata_value,
    )

    if not _check_mask_data_sufficient(
        feature_mask, config.data_build.feature_nodata_maximum_fraction
    ):
        return False

    if config.data_build.response_background_values:
        if np.all(
            np.in1d(local_response, config.data_build.response_background_values)
        ):
            return False

    # Final check (propogate mask forward), and return
    local_feature[feature_mask, :] = np.nan
    local_response[response_mask, :] = np.nan

    features, responses = _open_temporary_features_responses_data_files(
        config, local_feature.shape[-1], local_response.shape[-1], read_type="r+"
    )
    features[sample_index, ...] = local_feature
    responses[sample_index, ...] = local_response
    del features, responses

    return True


################### File/Dataset Opening Functions ##############################


def _open_temporary_features_responses_data_files(
    config: configs.Config, num_features: int, num_responses: int, read_type: str = "w+"
) -> Tuple[np.array, np.array]:
    basename = data_core.get_memmap_basename(config)
    shape = [
        config.data_build.max_samples,
        config.data_build.window_radius * 2,
        config.data_build.window_radius * 2,
    ]
    shape_features = tuple(shape + [num_features])
    shape_responses = tuple(shape + [num_responses])

    features_filepath = data_core.get_temporary_features_filepath(config)
    responses_filepath = data_core.get_temporary_responses_filepath(config)

    _logger.debug(
        "Create temporary munged features data file with shape {} at {}".format(
            shape_features, features_filepath
        )
    )
    features = np.memmap(
        features_filepath, dtype=np.float32, mode=read_type, shape=shape_features
    )

    _logger.debug(
        "Create temporary munged responses data file with shape {} at {}".format(
            shape_responses, responses_filepath
        )
    )
    responses = np.memmap(
        responses_filepath, dtype=np.float32, mode=read_type, shape=shape_responses
    )

    # features_dataset = h5py.File(features_filepath, read_type)
    # responses_dataset = h5py.File(responses_filepath, read_type)

    # if (read_type == 'w'):
    #    features = features_dataset.create_dataset('features', shape_features, dtype=np.float32, chunks=True)
    #    responses = responses_dataset.create_dataset('responses', shape_responses, dtype=np.float32, chunks=True)
    # else:
    #    features = features_dataset['features']
    #    responses = responses_dataset['responses']
    return features, responses


def _create_temporary_weights_data_files(
    config: configs.Config, num_samples: int
) -> np.array:
    weights_filepath = data_core.get_temporary_weights_filepath(config)
    _logger.debug(
        "Create temporary munged weights data file at {}".format(weights_filepath)
    )
    shape = tuple(
        [
            num_samples,
            config.data_build.window_radius * 2,
            config.data_build.window_radius * 2,
            1,
        ]
    )
    return np.memmap(weights_filepath, dtype=np.float32, mode="w+", shape=shape)


def load_built_data_files(
    config: configs.Config, writeable: bool = False
) -> Tuple[List[np.array], List[np.array], List[np.array]]:
    _logger.debug("Loading built data files with writeable == {}".format(writeable))
    feature_files = data_core.get_built_features_filepaths(config)
    response_files = data_core.get_built_responses_filepaths(config)
    weight_files = data_core.get_built_weights_filepaths(config)
    mode = "r+" if writeable else "r"

    features = [np.load(feature_file, mmap_mode=mode) for feature_file in feature_files]
    responses = [
        np.load(response_file, mmap_mode=mode) for response_file in response_files
    ]
    weights = [np.load(weight_file, mmap_mode=mode) for weight_file in weight_files]
    _logger.debug("Built data files loaded")
    return features, responses, weights


################### Save/remove Functions ##############################


def _save_built_data_files(
    features_munged: np.array,
    responses_munged: np.array,
    weights_munged: np.array,
    config: configs.Config,
) -> None:
    _logger.debug("Create fold assignments")
    fold_assignments = np.zeros(features_munged.shape[0]).astype(int)
    for f in range(0, config.data_build.number_folds):
        idx_start = int(
            round(f / config.data_build.number_folds * len(fold_assignments))
        )
        idx_finish = int(
            round((f + 1) / config.data_build.number_folds * len(fold_assignments))
        )
        fold_assignments[idx_start:idx_finish] = f

    _logger.debug("Save features to memmapped arrays separated by folds")
    features_filepaths = data_core.get_built_features_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug("Save features fold {}".format(idx_fold))
        np.save(
            features_filepaths[idx_fold],
            features_munged[fold_assignments == idx_fold, ...],
        )

    _logger.debug("Save responses to memmapped arrays separated by folds")
    responses_filepaths = data_core.get_built_responses_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug("Save responses fold {}".format(idx_fold))
        np.save(
            responses_filepaths[idx_fold],
            responses_munged[fold_assignments == idx_fold, ...],
        )

    _logger.debug("Save weights to memmapped arrays separated by folds")
    weights_filepaths = data_core.get_built_weights_filepaths(config)
    for idx_fold in range(config.data_build.number_folds):
        _logger.debug("Save weights fold {}".format(idx_fold))
        np.save(
            weights_filepaths[idx_fold],
            weights_munged[fold_assignments == idx_fold, ...],
        )


def _save_built_data_config_sections_to_verify_successful(
    config: configs.Config
) -> None:
    filepath = data_core.get_built_data_config_filepath(config)
    _logger.debug("Saving built data config sections to {}".format(filepath))
    configs.save_config_to_file(
        config, filepath, include_sections=["raw_files", "data_build"]
    )


def _remove_temporary_data_files(config: configs.Config) -> None:
    _logger.debug("Remove temporary munge files")
    if os.path.exists(data_core.get_temporary_features_filepath(config)):
        os.remove(data_core.get_temporary_features_filepath(config))
    if os.path.exists(data_core.get_temporary_responses_filepath(config)):
        os.remove(data_core.get_temporary_responses_filepath(config))
    if os.path.exists(data_core.get_temporary_weights_filepath(config)):
        os.remove(data_core.get_temporary_weights_filepath(config))


################### Array Resizing Functions ##############################


def _resize_munged_features(
    features_munged: np.array, num_samples: int, config: configs.Config
) -> np.array:
    _logger.debug(
        "Resize memmapped features array with out-of-memory methods; original features shape {}".format(
            features_munged.shape
        )
    )
    new_features_shape = tuple([num_samples] + list(features_munged.shape[1:]))
    _logger.debug("Delete in-memory data to force data dump")
    del features_munged
    _logger.debug(
        "Reload data from memmap files with modified sizes; new features shape {}".format(
            new_features_shape
        )
    )
    features_munged = np.memmap(
        data_core.get_temporary_features_filepath(config),
        dtype=np.float32,
        mode="r+",
        shape=new_features_shape,
    )
    return features_munged


def _resize_munged_responses(
    responses_munged: np.array, num_samples: int, config: configs.Config
) -> np.array:
    _logger.debug(
        "Resize memmapped responses array with out-of-memory methods; original responses shape {}".format(
            responses_munged.shape
        )
    )
    new_responses_shape = tuple([num_samples] + list(responses_munged.shape[1:]))
    _logger.debug("Delete in-memory data to force data dump")
    del responses_munged
    _logger.debug(
        "Reload data from memmap files with modified sizes; new responses shape {}".format(
            new_responses_shape
        )
    )
    responses_munged = np.memmap(
        data_core.get_temporary_responses_filepath(config),
        dtype=np.float32,
        mode="r+",
        shape=new_responses_shape,
    )
    return responses_munged


################### Verification Functions ##############################


def _check_mask_data_sufficient(mask: np.array, max_nodata_fraction: float) -> bool:
    if mask is not None:
        nodata_fraction = np.sum(mask) / np.prod(mask.shape)
        if nodata_fraction <= max_nodata_fraction:
            _logger.debug(
                "Data mask has sufficient data, missing data proportion: {}".format(
                    nodata_fraction
                )
            )
            return True
        else:
            _logger.debug(
                "Data mask has insufficient data, missing data proportion: {}".format(
                    nodata_fraction
                )
            )
            return False
    else:
        _logger.debug("Data mask is None")
        return False


def _is_vector_file(boundary_filepath: str) -> bool:
    return (
        str(boundary_filepath.split(".")[-1]).lower() in sections.VECTORIZED_FILENAMES
    )


def check_built_data_files_exist(config: configs.Config) -> bool:
    filepaths = (
        data_core.get_built_features_filepaths(config)
        + data_core.get_built_responses_filepaths(config)
        + data_core.get_built_weights_filepaths(config)
    )
    missing_files = [filepath for filepath in filepaths if not os.path.exists(filepath)]
    if not missing_files:
        _logger.debug(
            "Built data files found at paths: {}".format(", ".join(filepaths))
        )
    else:
        _logger.info(
            "Built data files were not found at paths: {}".format(
                ", ".join(missing_files)
            )
        )
    return not missing_files


################### Logging Functions ##############################


def _log_munged_data_information(
    features_munged: np.array = None,
    responses_munged: np.array = None,
    weights_munged: np.array = None,
) -> None:
    if features_munged is not None:
        _logger.info("Munged features shape: {}".format(features_munged.shape))
    if responses_munged is not None:
        _logger.info("Munged responses shape: {}".format(responses_munged.shape))
    if weights_munged is not None:
        _logger.info("Munged weights shape: {}".format(weights_munged.shape))
