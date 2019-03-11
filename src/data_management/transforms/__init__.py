import numpy as np
from sklearn.externals import joblib
import sklearn.preprocessing

from src.utils import logger


_logger = logger.get_child_logger(__name__)


class BaseGlobalTransformer(object):
    """
    Transformers handle the process of transforming data prior to fitting or predicting using the neural network, as
    well as inverse transforming the data for applications or review afterwards. In this case, we use readily available
    scalers from the scikit-learn package to handle the nitty-gritty of the transform and inverse transform, and we use
    the Transformer class to handle the nitty-gritty of reshaping and otherwise handling the image arrays.
    """
    nodata_value = None
    savename = None
    _savename_prefix = 'transformer'
    scaler_name = None

    def __init__(self, nodata_value):
        self.nodata_value = nodata_value
        self.savename = self._savename_prefix + '_' + self.scaler_name
        self.is_fitted = False

    def fit(self, image_array):
        assert self.is_fitted is False, 'Transformer has already been fit to data'
        # Reshape to (num_samples, num_features)
        image_array = self._reshape_image_array(image_array)
        image_array[image_array == self.nodata_value] = np.nan
        self._fit(image_array)
        self.is_fitted = True

    def _fit(self, image_array):
        raise NotImplementedError

    def inverse_transform(self, image_array):
        shape = image_array.shape

        bad_dat = image_array == self.nodata_value
        image_array[bad_dat] = np.nan

        # Reshape to (num_samples, num_features)
        image_array = self._reshape_image_array(image_array)

        image_array = self._inverse_transform(image_array)
        image_array = image_array.reshape(shape)
        image_array[bad_dat] = self.nodata_value

        return image_array

    def _inverse_transform(self, image_array):
        raise NotImplementedError

    def transform(self, image_array):
        shape = image_array.shape
        # Reshape to (num_samples, num_features)
        image_array = self._reshape_image_array(image_array)

        # convert nodata values to nans temporarily
        bad_dat = image_array == self.nodata_value
        image_array[bad_dat] = np.nan

        image_array = self._transform(image_array)
        image_array = image_array.reshape(shape)
        image_array[bad_dat] = self.nodata_value

        num_conflicts = np.sum(image_array[np.isfinite(image_array)] != self.nodata_value)
        if num_conflicts > 0:
            _logger.warn('{} values in transformed data are not equal to nodata value'
                         .format(num_conflicts))
        image_array[~np.isfinite(image_array)] = self.nodata_value
        return image_array

    def _transform(self, image_array):
        raise NotImplementedError

    def _reshape_image_array(self, image_array):
        # The second dimension is image_array.shape[-1] which is the num_channels, so the first dimension is
        # image width x image height
        return image_array.reshape(-1, image_array.shape[-1])

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


class BaseSklearnTransformer(BaseGlobalTransformer):
    scaler = None

    def __init__(self, nodata_value):
        self.scaler_name = 'sklearn_' + self.scaler.__class__.__name__
        super().__init__(nodata_value)

    def _fit(self, image_array):
        self.scaler.fit(image_array)

    def _inverse_transform(self, image_array):
        return self.scaler.inverse_transform(image_array)

    def _transform(self, image_array):
        return self.scaler.transform(image_array)

    def save(self):
        joblib.dump(self.scaler, self.savename)

    def load(self):
        self.scaler = joblib.load(self.savename)


class ConstantTransformer(BaseGlobalTransformer):
    constant_scaler = None
    constant_offset = None

    def __init__(self, nodata_value, constant_scaler, constant_offset=None):
        self.constant_scaler = constant_scaler
        self.constant_offset = constant_offset

        self.scaler_name = 'ConstantScaler'
        super().__init__(nodata_value)

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
        npzf = np.load(self.savename + '.npz')
        self.constant_scaler = npzf['constant_scaler']
        self.constant_offset = npzf['constant_offset']


class StandardTransformer(BaseSklearnTransformer):

    def __init__(self, nodata_value):
        self.scaler = sklearn.preprocessing.StandardScaler(copy=True)
        super().__init__(nodata_value)


class MinMaxTransformer(BaseSklearnTransformer):

    def __init__(self, nodata_value, feature_range=(-1, 1)):
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range, copy=True)
        super().__init__(nodata_value)


class RobustTransformer(BaseSklearnTransformer):

    def __init__(self, nodata_value, quantile_range=(10.0, 90.0)):
        self.scaler = sklearn.preprocessing.RobustScaler(quantile_range=quantile_range, copy=True)
        super().__init__(nodata_value)


class PowerTransformer(BaseSklearnTransformer):

    def __init__(self, nodata_value, method='box-cox'):
        self.scaler = sklearn.preprocessing.PowerTransformer(method=method, copy=True)
        super().__init__(nodata_value)


class QuantileUniformTransformer(BaseSklearnTransformer):

    def __init__(self, nodata_value, output_distribution='uniform'):
        self.scaler = sklearn.preprocessing.QuantileTransformer(output_distribution=output_distribution, copy=True)
        super().__init__(nodata_value)
