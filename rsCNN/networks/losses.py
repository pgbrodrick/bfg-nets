from typing import Callable

import keras


def cropped_loss(loss_type: str, outer_width: int, inner_width: int, weighted: bool = True) -> Callable:
    """ Loss function with optional per-pixel weighting
        and edge trimming options.

    Arguments:
    loss_type - str
      The type of loss function to implement.  Currently enabled:
        mae
        mse
        rmse
        categorical_crossentropy
    outer_width - int
      The width of the input image.
    inner_width - int
      The width of the input image to use in the loss function


    Keyword Arguments:
    weighted - bool
      Tells whether the training y has weights as the last dimension
      to apply to the loss function.
    """

    def _cropped_loss(y_true, y_pred):
        if outer_width != inner_width:
            buffer = int(round((outer_width-inner_width) / 2))
            y_true = y_true[:, buffer:-buffer, buffer:-buffer, :]
            y_pred = y_pred[:, buffer:-buffer, buffer:-buffer, :]

        if weighted:
            weights = y_true[..., -1]
        y_true = y_true[..., :-1]

        if (loss_type == 'cc' or loss_type == 'categorical_crossentropy'):
            loss = keras.backend.categorical_crossentropy(y_true, y_pred)
        elif (loss_type == 'mae' or loss_type == 'mean_absolute_error'):
            loss = keras.backend.mean(keras.backend.abs(y_true - y_pred))
        elif (loss_type == 'mse' or loss_type == 'mean_squared_error'):
            loss = keras.backend.mean(keras.backend.pow(y_true - y_pred, 2))
        elif (loss_type == 'rmse' or loss_type == 'root_mean_squared_eror'):
            loss = keras.backend.pow(keras.backend.mean(keras.backend.pow(y_true - y_pred, 2)), 0.5)
        else:
            raise NotImplementedError('Unknown loss function')

        # TODO: check that this after the fact weight multiplication works properly
        if weighted:
            loss = loss * weights

        return loss
    return _cropped_loss
