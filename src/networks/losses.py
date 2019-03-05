import keras






# TODO: - delete if you're okay with the update
#def cropped_mse(loss_buffer_constant):
#    def _cropped_mse(y_true, y_pred):
#        # Need to reference y_pred shape, y_true shape will be tuple of Nones during compilation
#        buffer_x = int(loss_buffer_constant * keras.backend.int_shape(y_pred)[1])
#        buffer_y = int(loss_buffer_constant * keras.backend.int_shape(y_pred)[2])
#        y_true = y_true[:, buffer_x:-buffer_x, buffer_y:-buffer_y, :]
#        y_pred = y_pred[:, buffer_x:-buffer_x, buffer_y:-buffer_y, :]
#        return keras.losses.mean_squared_error(y_true, y_pred)
#    return _cropped_mse




def cropped_loss(outer_width,inner_width,loss_type,weighted=False):
    """ Loss function with optional per-pixel weighting
        and edge trimming options.

    Arguments:
    outer_width - int
      The width of the input image.
    inner_width - int
      The width of the input image to use in the loss function
    loss_type - str
      The type of loss function to implement.  Currently enabled:
        mae
        mse
        rmse
        categorical_crossentropy


    Keyword Arguments:
    weighted - bool
      Tells whether the training y has weights as the last dimension
      to apply to the loss function.
    """

    def _cropped_cc(y_true, y_pred):
        if (outer_width != inner_width):
          buffer = rint((outer_width-inner_width) / 2)
          y_true = y_true[:, buffer:-buffer, buffer:-buffer, :]
          y_pred = y_pred[:, buffer:-buffer, buffer:-buffer, :]

          if (loss_type == 'categorical_crossentropy'):
            loss = keras.backend.categorical_crossentropy(y_true[...,:-1],y_pred)
          elif (loss_type == 'mae'):
            loss = keras.backend.mean(keras.backend.abs(y_true[...,:-1]-y_pred))
          elif (loss_type == 'mse'):
            loss = keras.backend.mean(keras.backend.power(y_true[...,:-1]-y_pred,2))
          elif (loss_type == 'rmse'):
            loss = keras.backend.power(keras.backend.mean(keras.backend.power(y_true[...,:-1]-y_pred,2)),0.5)
          else:
            raise NotImplementedError('Unknown loss function')

          if (weighted):
            loss = loss * y_true[...,-1]
        else:
          return loss
    return _cropped_cc
















