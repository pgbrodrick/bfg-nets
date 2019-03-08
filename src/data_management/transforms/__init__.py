import numpy as np
from sklearn.externals import joblib
import sklearn.preprocessing

from src.utils import logging


_logger = logging.get_child_logger(__name__)


class BaseGlobalTransformer(object):
    """
    Transformers handle the process of transforming data prior to fitting or predicting using the neural network, as
    well as inverse transforming the data for applications or review afterwards. In this case, we use readily available
    scalers from the scikit-learn package to handle the nitty-gritty of the transform and inverse transform, and we use
    the Transformer class to handle the nitty-gritty of reshaping and otherwise handling the image arrays.
    """
    is_fitted = False
    scaler = None
    scaler_name = None

    def __init__(nodata_value,savename=None,):
        self.nodata_value = nodata_value
        self.savename = savename


    def fit(self, image_array):
        assert self.is_fitted is False, 'Transformer has already been fit to data'
        image_array = self._reshape_image_array(image_array)  # Needs to be reshaped for (num_samples, num_features)
        image_array[image_array == self.nodata_value] = np.nan
        self.scaler.fit(image_array)
        self.is_fitted = True

    def inverse_transform(self, image_array):
        shape = image_array.shape

        bad_dat = image_array == self.nodata_value
        image_array[bad_dat] = np.nan

        image_array = self._reshape_image_array(image_array)  # Needs to be reshaped for (num_samples, num_features)

        image_array = self.scaler.inverse_transform(image_array).reshape(shape)
        image_array[bad_dat] = self.nodata_value

        return image_array

    def transform(self, image_array):
        shape = image_array.shape
        image_array = self._reshape_image_array(image_array)  # Needs to be reshaped for (num_samples, num_features)

        # convert nodata values to nans temporarily
        bad_dat = image_array == self.nodata_value
        image_array[bad_dat] = np.nan

        image_array = self.scaler.transform(image_array).reshape(shape)
        image_array[bad_dat] = self.nodata_value

        num_conflicts = np.sum(image_array[np.isfinite(image_array)] != self.nodata_value)
        if num_conflicts > 0:
            _logger.warn('{} values in transformed data are not equal to nodata value'
                         .format(num_conflicts))
        image_array[~np.isfinite(image_array)] = self.nodata_value
        return image_array

    def _reshape_image_array(self, image_array):
        return image_array.reshape(-1, image_array.shape[-1])

    def save(self):
        if ('sklearn' in self.scaler_name):
          joblib.dump(self.scaler,self.savename)
        else:
          raise NotImplementedError('Need to write code to load/save transformers')

    def load_transformer():
        if ('sklearn' in self.scaler_name):
          self.scaler = joblib.load(self.savename)
        else: 
          raise NotImplementedError('Need to write code to load/save transformers')




class StandardTransformer(BaseTransformer):

    def __init__(self, feature_range=(-1, 1)):
        self.scaler = sklearn.preprocessing.StandardScaler(copy=True)
        self.scaler_name = 'sklearn_StandardScaler'
        super().__init__(self)


class MinMaxTransformer(BaseTransformer):

    def __init__(self, feature_range=(-1, 1)):
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range, copy=True)
        self.scaler_name = 'sklearn_MinMaxScaler'
        super().__init__(self)


class RobustTransformer(BaseTransformer):

    def __init__(self, quantile_range=(10.0, 90.0)):
        self.scaler = sklearn.preprocessing.RobustScaler(quantile_range=quantile_range, copy=True)
        self.scaler_name = 'sklearn_RobustScaler'
        super().__init__(self)


class PowerTransformer(BaseTransformer):

    def __init__(self, method='box-cox'):
        self.scaler = sklearn.preprocessing.PowerTransformer(method=method, copy=True)
        self.scaler_name = 'sklearn_PowerTransformer'
        super().__init__(self)


class QuantileUniformTransformer(BaseTransformer):

    def __init__(self, output_distribution='uniform'):
        self.scaler = sklearn.preprocessing.QuantileTransformer(output_distribution=output_distribution, copy=True)
        self.scaler_name = 'sklearn_QuantileTransformer'
        super().__init__(self)



def build_scaling(config,features,responses,fold_assignments):

    
    if (global_feature_scaling















