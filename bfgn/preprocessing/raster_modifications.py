from typing import List, Tuple

import gdal
import numpy as np
from scipy import signal


def add_edges_to_binary_image(
    input_file: str, output_file: str, binary_values: List = [0, 1]
):

    input_dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    raw_data = input_dataset.ReadAsArray()
    binary_data = (raw_data == binary_values[0]).astype(int)
    del raw_data
    binary_values = [0, 1]

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    outDataset = driver.Create(
        output_file,
        input_dataset.RasterXSize,
        input_dataset.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW"],
    )

    outDataset.SetProjection(input_dataset.GetProjection())
    outDataset.SetGeoTransform(input_dataset.GetGeoTransform())

    binary_data[
        np.logical_and(
            binary_data == binary_values[0],
            signal.convolve2d(
                (binary_data == binary_values[1]).astype(int),
                np.ones((3, 3)),
                mode="same",
                fillvalue=0,
            ),
        ).astype(bool)
    ] = 2

    outDataset.GetRasterBand(1).WriteArray(binary_data, 0, 0)
    del outDataset
