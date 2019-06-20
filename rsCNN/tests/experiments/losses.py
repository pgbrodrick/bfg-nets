from rsCNN.experiments import losses


def test_get_cropped_loss_function_runs_without_error():
    assert losses.get_cropped_loss_function('rmse', 4, 2, True)
