import keras.backend as K
import numpy as np

from bfgn.experiments import losses


def test_get_cropped_loss_function_runs_without_error() -> None:
    assert losses.get_cropped_loss_function('rmse', 4, 2)


def test_get_cropped_loss_function_window_size_correct() -> None:
    num_samples = 11
    outer = 10
    inner = 7
    calculate_cropped_loss = losses.get_cropped_loss_function('mae', outer, inner)
    y_true = np.ones((num_samples, outer, outer, 1))
    y_pred = np.zeros((num_samples, outer, outer, 1))
    loss = K.get_value(calculate_cropped_loss(y_true, y_pred))
    actual_shape = loss.shape
    expected_size = 2 * (outer - inner)
    expected_shape = (num_samples, expected_size, expected_size)
    assert actual_shape == expected_shape
