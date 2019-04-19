from typing import List

import os

import matplotlib.pyplot as plt
import numpy as np


def walk_directories_for_model_histories(directories: List[str]) -> List[str]:
    paths_histories = list()
    for directory in directories:
        for path, dirs, files in os.walk(directory):
            for file_ in files:
                if file_ == 'history.pkl':
                    paths_histories.append(os.path.join(path, file_))
    return paths_histories


def plot_model_loss_comparison(model_histories: List[dict]) -> List[plt.Figure]:
    fig, axes = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
    for history in sorted(model_histories, key=lambda x: x['model_name']):
        axes[0].plot(history['loss'], label=history['model_name'])
        axes[1].plot(history['val_loss'])
    for ax in axes:
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
    fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0, -0.1, 1, 1), bbox_transform=plt.gcf().transFigure)
    axes[0].set_title('Training loss')
    axes[1].set_title('Validation loss')
    return [fig]


def plot_model_timing_comparison(model_histories: List[dict]) -> List[plt.Figure]:
    # TODO:  add validation/test timings
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list()
    timings = list()
    for history in sorted(model_histories, key=lambda x: x['model_name']):
        labels.append(history['model_name'])
        timings.append((history['train_finish'] - history['train_start']).minutes)
    ax.barh(np.arange(len(timings)), timings, tick_label=labels)
    ax.set_xlabel('Minutes')
    ax.set_title('Training times')
    return [fig]
