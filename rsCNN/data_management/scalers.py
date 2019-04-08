import sys

import numpy as np
import os
from sklearn.externals import joblib
import sklearn.preprocessing

from rsCNN.utils import logger


_logger = logger.get_child_logger(__name__)


def get_scaler(scaler_name, scaler_options):
    return getattr(sys.modules[__name__], scaler_name)(**scaler_options)


class BaseGlobalScaler(object):
    """
    Scalers handle the process of transforming data prior to fitting or predicting using the neural network, as
    well as inverse transforming the data for applications or review afterwards. In this case, we use readily available
    scalers from the scikit-learn package to handle the nitty-gritty of the transform and inverse transform, and we use
    the Scaler class to handle the nitty-gritty of reshaping and otherwise handling the image arrays.
    """
    nodata_value = None
    savename = None
    scaler_name = None

    def __init__(self, nodata_value=None, savename_base=None):
        """
        :param savename_base: the directory and optionally filename prefix for saving data
        """
        self.nodata_value = nodata_value
        if (savename_base is not None):
            self.savename = savename_base + self.scaler_name
        self.is_fitted = False

    def fit(self, image_array):
        assert self.is_fitted is False, 'Scaler has already been fit to data'
        image_array[image_array == self.nodata_value] = np.nan
        self._fit(image_array)
        self.is_fitted = True

    def _fit(self, image_array):
        raise NotImplementedError

    def inverse_transform(self, image_array):
        # User can overwrite image_array values with nodata_value if input data is missing
        image_array[image_array == self.nodata_value] = np.nan
        image_array = self._inverse_transform(image_array)
        image_array[~np.isfinite(image_array)] = self.nodata_value
        return image_array

    def _inverse_transform(self, image_array):
        raise NotImplementedError

    def transform(self, image_array):
        # convert nodata values to nans temporarily
        image_array[image_array == self.nodata_value] = np.nan
        image_array = self._transform(image_array)
        num_conflicts = np.sum(image_array == self.nodata_value)
        if num_conflicts > 0:
            _logger.warn('{} values in transformed data are equal to nodata value'.format(num_conflicts))
        image_array[~np.isfinite(image_array)] = self.nodata_value
        return image_array

    def _transform(self, image_array):
        raise NotImplementedError

    def fit_transform(self, image_array):
        self.fit(image_array)
        self.transform(image_array)
        return image_array

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class BaseSklearnScaler(BaseGlobalScaler):
    scaler = None

    def __init__(self, nodata_value, savename_base):
        self.scaler_name = 'sklearn_' + self.scaler.__class__.__name__
        super().__init__(nodata_value, savename_base)

    def _fit(self, image_array):
        # Reshape to (num_samples, num_features)
        image_array = self._reshape_image_array(image_array)
        self.scaler.fit(image_array)

    def _inverse_transform(self, image_array):
        # Reshape to (num_samples, num_features) for sklearn
        shape = image_array.shape
        image_array = self._reshape_image_array(image_array)
        image_array = self.scaler.inverse_transform(image_array)
        return image_array.reshape(shape)

    def _transform(self, image_array):
        # Reshape to (num_samples, num_features) for sklearn
        shape = image_array.shape
        image_array = self._reshape_image_array(image_array)
        image_array = self.scaler.transform(image_array)
        return image_array.reshape(shape)

    def _reshape_image_array(self, image_array):
        # The second dimension is image_array.shape[-1] which is the num_channels, so the first dimension is
        # image width x image height
        return image_array.reshape(-1, image_array.shape[-1])

    def save(self):
        joblib.dump(self.scaler, self.savename)

    def load(self):
        if (os.path.isfile(self.savename)):
            self.scaler = joblib.load(self.savename)
            self.is_fitted = True


class NullScaler(BaseGlobalScaler):

    def __init__(self, nodata_value, savename_base):
        self.scaler_name = 'NullScaler'
        super().__init__(nodata_value, savename_base)

    def _fit(self, image_array):
        return image_array

    def _inverse_transform(self, image_array):
        return image_array

    def _transform(self, image_array):
        return image_array

    def save(self):
        pass

    def load(self):
        self.is_fitted = True


class ConstantScaler(BaseGlobalScaler):
    constant_scaler = None
    constant_offset = None

    def __init__(self, nodata_value, savename_base, constant_scaler=None, constant_offset=None):
        self.constant_scaler = constant_scaler
        self.constant_offset = constant_offset

        self.scaler_name = 'ConstantScaler'
        super().__init__(nodata_value, savename_base)

    def _fit(self, image_array):
        pass

    def _inverse_transform(self, image_array):
        idx_valid = image_array != self.nodata_value
        image_array[idx_valid] = (image_array[idx_valid] - self.constant_offset) * self.constant_scaler
        return image_array

    def _transform(self, image_array):
        idx_valid = image_array != self.nodata_value
        image_array[idx_valid] = image_array[idx_valid] / self.constant_scaler + self.constant_offset
        return image_array

    def save(self):
        np.savez(self.savename + '.npz', constant_scaler=self.constant_scaler, constant_offset=self.constant_offset)

    def load(self):
        if (os.path.isfile(self.savename + '.npz')):
            npzf = np.load(self.savename + '.npz')
            self.constant_scaler = npzf['constant_scaler']
            self.constant_offset = npzf['constant_offset']
            self.is_fitted = True


class StandardScaler(BaseSklearnScaler):

    def __init__(self, nodata_value, savename_base):
        self.scaler = sklearn.preprocessing.StandardScaler(copy=True)
        super().__init__(nodata_value, savename_base)


class MinMaxScaler(BaseSklearnScaler):

    def __init__(self, nodata_value, savename_base, feature_range=(-1, 1)):
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range, copy=True)
        super().__init__(nodata_value, savename_base)


class RobustScaler(BaseSklearnScaler):

    def __init__(self, nodata_value, savename_base, quantile_range=(10.0, 90.0)):
        self.scaler = sklearn.preprocessing.RobustScaler(quantile_range=quantile_range, copy=True)
        super().__init__(nodata_value, savename_base)


class PowerScaler(BaseSklearnScaler):

    def __init__(self, nodata_value, savename_base, method='box-cox'):
        self.scaler = sklearn.preprocessing.PowerTransformer(method=method, copy=True)
        super().__init__(nodata_value, savename_base)


class QuantileUniformScaler(BaseSklearnScaler):

    def __init__(self, nodata_value, savename_base, output_distribution='uniform'):
        self.scaler = sklearn.preprocessing.QuantileTransformer(output_distribution=output_distribution, copy=True)
        super().__init__(nodata_value, savename_base)
