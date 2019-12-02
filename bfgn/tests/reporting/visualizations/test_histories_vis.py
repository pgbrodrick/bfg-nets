import datetime

import matplotlib.pyplot as plt

from bfgn.reporting.visualizations import histories


def test_plot_history_returns_empty_list_on_empty_history() -> None:
    assert histories.plot_history(dict()) == list()


def test_plot_history_returns_figures_with_valid_history() -> None:
    len_history = 10
    datetimes = [datetime.datetime.now()] * len_history
    valid_history = {
        "epoch_start": datetimes,
        "epoch_finish": datetimes,
        "train_start": datetime.datetime.now(),
        "loss": [1] * len_history,
        "val_loss": [1] * len_history,
        "lr": [1] * len_history,
    }
    plots = histories.plot_history(valid_history)
    assert type(plots) is list
    assert type(plots[0]) is plt.Figure


def test_plot_history_returns_figures_with_warnings() -> None:
    warnings_history = {"unused_key": 0}
    plots = histories.plot_history(warnings_history)
    assert type(plots) is list
    assert type(plots[0]) is plt.Figure
