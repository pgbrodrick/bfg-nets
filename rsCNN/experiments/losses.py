import logging
from typing import Callable

import keras


_logger = logging.getLogger(__name__)

_LOSS_TYPE_CC = ('cc', 'categorical_crossentropy')
_LOSS_TYPE_MAE = ('mae', 'mean_absolute_error')
_LOSS_TYPE_MSE = ('mse', 'mean_squared_error')
_LOSS_TYPE_RSME = ('rmse', 'root_mean_squared_error')


def get_valid_loss_methods():
    return [lm for lms in (_LOSS_TYPE_CC, _LOSS_TYPE_MAE, _LOSS_TYPE_MSE, _LOSS_TYPE_RSME) for lm in lms]


def get_cropped_loss_function(loss_method: str, outer_width: int, inner_width: int, weighted: bool = True) -> Callable:
    """ Creates a loss function callable with optional per-pixel weighting and edge-trimming.

    Arguments:
        loss_method: The loss calculate to implement, currently supports categorical_crossentropy or cc,
        mean_absolute_error or mae, mean_squared_error or mse, and root_mean_squared_error or rmse.
        outer_width: The full dimension (height or width) of the input image; e.g., 128 for a 128x128 image.
        inner_width: The full dimension (height or width) of the loss window to use. Must not be greater than 128 for a
        128x128 image, with generally better results at 25% to 50% of the full image size; i.e., 32 to 64 for a 128x128
        image.
        weighted: Whether the training response array has weights appended to the last axis to use in loss calculations.
        Keras does not have a simple way to pass weight values to loss functions, so a common work-around is to append
        weight values to the sample responses and reference those weights in the loss functions directly. The rsCNN
        package will automatically build and append weights if the configuration specifieds that the loss function
        should be weighted.

    Returns:
        Loss function callable to be passed to a Keras model.
    """
    _logger.debug('Creating {}weighted {} loss function with outer and inner window of {} and {}'.format(
        '' if weighted else 'un', loss_method, outer_width, inner_width
    ))

    def _cropped_loss(y_true, y_pred):
        if outer_width != inner_width:
            buffer = int(round((outer_width-inner_width) / 2))
            y_true = y_true[:, buffer:-buffer, buffer:-buffer, :]
            y_pred = y_pred[:, buffer:-buffer, buffer:-buffer, :]

        if weighted:
            weights = y_true[..., -1]
        y_true = y_true[..., :-1]

        if loss_method in _LOSS_TYPE_CC:
            loss = keras.backend.categorical_crossentropy(y_true, y_pred)
        elif loss_method in _LOSS_TYPE_MAE:
            loss = keras.backend.mean(keras.backend.abs(y_true - y_pred))
        elif loss_method in _LOSS_TYPE_MSE:
            loss = keras.backend.mean(keras.backend.pow(y_true - y_pred, 2))
        elif loss_method in _LOSS_TYPE_RSME:
            loss = keras.backend.pow(keras.backend.mean(keras.backend.pow(y_true - y_pred, 2)), 0.5)

        if weighted:
            loss = loss * weights

        return loss
    return _cropped_loss
