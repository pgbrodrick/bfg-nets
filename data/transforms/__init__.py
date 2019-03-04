import numpy as np
import sklearn.preprocessing

from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)


def load_transformer(path):
    # TODO
    raise NotImplementedError('Need to write code to load/save transformers')


class BaseTransformer(object):
    """
    Transformers handle the process of transforming data prior to fitting or predicting using the neural network, as
    well as inverse transforming the data for applications or review afterwards. In this case, we use readily available
    scalers from the scikit-learn package to handle the nitty-gritty of the transform and inverse transform, and we use
    the Transformer class to handle the nitty-gritty of reshaping and otherwise handling the image arrays.
    """
    # TODO:
    """
    Phil, I'm not quite sure if these components are necessary, but I assume we want this in some form? That is, we're
    saving models and model metadata and we'd hypothetically want to load up the model and run it on new data. That 
    means that we also need to save the transformer information so that we can do the correct operations on the new 
    data, right? Assuming this is useful and the right place/structure, you could add your mean/std/other 
    transformations as objects below, where you'd just need to save the relevant parameters. I'll be able to add a 
    method to save the sklearn transformers when we're ready.
    """
    scaler = None
    encoding_missing = None
    is_fitted = False

    def fit(self, image_array, encoding_missing):
        assert self.is_fitted is False, 'Transformer has already been fit to data'
        self.encoding_missing = encoding_missing
        image_array = self._reshape_image_array(image_array)  # Needs to be reshaped for (num_samples, num_features)
        self.scaler.fit(image_array)
        self.is_fitted = True

    def inverse_transform(self, image_array):
        shape = image_array.shape
        image_array = self._reshape_image_array(image_array)  # Needs to be reshaped for (num_samples, num_features)
        return self.scaler.inverse_transform(image_array).reshape(shape)

    def transform(self, image_array):
        shape = image_array.shape
        image_array = self._reshape_image_array(image_array)  # Needs to be reshaped for (num_samples, num_features)
        image_array = self.scaler.transform(image_array).reshape(shape)
        num_conflicts = np.sum(image_array[np.isfinite(image_array)] <= self.encoding_missing)
        if num_conflicts > 0:
            _logger.warn('{} values in transformed data are less than or equal to missing data encoding value'
                         .format(num_conflicts))
        image_array[~np.isfinite(image_array)] = self.encoding_missing
        return image_array

    def _reshape_image_array(self, image_array):
        return image_array.reshape(-1, image_array.shape[-1])

    def save(self):
        # TODO
        raise NotImplementedError('Need to write code to load/save transformers')


class MinMaxTransformer(BaseTransformer):

    def __init__(self, feature_range=(-1, 1)):
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range, copy=True)


class RobustTransformer(BaseTransformer):

    def __init__(self, quantile_range=(10.0, 90.0)):
        self.scaler = sklearn.preprocessing.RobustScaler(quantile_range=quantile_range, copy=True)


class PowerTransformer(BaseTransformer):

    def __init__(self, method='box-cox'):
        self.scaler = sklearn.preprocessing.PowerTransformer(method=method, copy=True)


class QuantileUniformTransformer(BaseTransformer):

    def __init__(self, output_distribution='uniform'):
        self.scaler = sklearn.preprocessing.QuantileTransformer(output_distribution=output_distribution, copy=True)
