import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np

_logger = logging.getLogger(__name__)


def plot_history(history: dict) -> [plt.Figure]:
    if not history:
        _logger.debug("History not plotted; unable to plot empty history object.")
        return list()

    fig, axes = plt.subplots(figsize=(12, 10), nrows=2, ncols=2)

    # Epoch times and delays
    ax = axes[0, 0]
    if "epoch_start" in history and "epoch_finish" in history:
        epoch_time = [
            (finish - start).seconds
            for start, finish in zip(history["epoch_start"], history["epoch_finish"])
        ]
        epoch_delay = [
            (start - finish).seconds
            for start, finish in zip(
                history["epoch_start"][1:], history["epoch_finish"][:-1]
            )
        ]
        ax.plot(epoch_time, c="black", label="Epoch time")
        ax.plot(epoch_delay, "--", c="blue", label="Epoch delay")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds")
        ax.legend()
    else:
        _plot_warning_message(ax, "epoch timings", ["epoch_start", "epoch_finish"])

    # Epoch times different view
    ax = axes[0, 1]
    if "train_start" in history and "epoch_finish" in history:
        minutes_elapsed_per_epoch = np.array(
            [
                (dt - history["train_start"]).seconds / 60
                for dt in history["epoch_finish"]
            ]
        )
        minutes_elapsed = range(0, int(1 + max(minutes_elapsed_per_epoch)))
        cumulative_epochs = [
            sum(minutes_elapsed_per_epoch < minutes) for minutes in minutes_elapsed
        ]
        ax.plot(cumulative_epochs, c="black")
        ax.set_xlabel("Minutes elapsed since training started")
        ax.set_ylabel("Cumulative epochs completed")
    else:
        _plot_warning_message(ax, "epochs completed", ["train_start", "epoch_finish"])

    # Loss
    ax = axes[1, 0]
    if "loss" in history:
        ax.plot(history["loss"][-160:], c="black", label="Training loss")
        if "val_loss" in history:
            ax.plot(history["val_loss"][-160:], "--", c="blue", label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    else:
        _plot_warning_message(ax, "model loss", ["loss"])

    # Learning rate
    ax = axes[1, 1]
    if "lr" in history:
        ax.plot(history["lr"][-160:], c="black")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
    else:
        _plot_warning_message(ax, "learning rate", ["lr"])

    # Add figure title
    plt.suptitle("Model Training History")
    return [fig]


def _plot_warning_message(ax: plt.Axes, label: str, keys: List[str]) -> None:
    text = "Unable to plot {}.\nRelevant information not available in history object.\nNeed history keys: {}".format(
        label, ", ".join(keys)
    )
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
