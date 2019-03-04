import keras


def cropped_mse(loss_buffer_constant):
    def _cropped_mse(y_true, y_pred):
        # Need to reference y_pred shape, y_true shape will be tuple of Nones during compilation
        buffer_x = int(loss_buffer_constant * keras.backend.int_shape(y_pred)[1])
        buffer_y = int(loss_buffer_constant * keras.backend.int_shape(y_pred)[2])
        y_true = y_true[:, buffer_x:-buffer_x, buffer_y:-buffer_y, :]
        y_pred = y_pred[:, buffer_x:-buffer_x, buffer_y:-buffer_y, :]
        return keras.losses.mean_squared_error(y_true, y_pred)
    return _cropped_mse
