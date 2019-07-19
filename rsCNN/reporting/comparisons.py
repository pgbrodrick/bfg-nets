import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from rsCNN.experiments import histories


plt.switch_backend('Agg')  # Needed for remote server plotting


def create_model_comparison_report(
        filepath_out: str,
        dirs_histories: List[str] = None,
        paths_histories: List[str] = None
) -> None:
    assert dirs_histories or paths_histories, \
        'Either provide a directory containing model histories or paths to model histories'
    if not paths_histories:
        paths_histories = list()
    if dirs_histories:
        paths_histories.extend(walk_directories_for_model_histories(dirs_histories))
        assert len(paths_histories) > 0, 'No model histories found to compare'
    model_histories = [histories.load_history(path_history) for path_history in paths_histories]
    with PdfPages(filepath_out) as pdf:
        _add_figures(plot_model_loss_comparison(model_histories), pdf)
        _add_figures(plot_model_timing_comparison(model_histories), pdf)


def _add_figures(figures: List[plt.Figure], pdf: PdfPages, tight: bool = True) -> None:
    for fig in figures:
        pdf.savefig(fig, bbox_inches='tight' if tight else None)


def walk_directories_for_model_histories(directories: List[str]) -> List[str]:
    paths_histories = list()
    for directory in directories:
        for path, dirs, files in os.walk(directory):
            for file_ in files:
                if file_ == histories.DEFAULT_FILENAME_HISTORY:
                    paths_histories.append(os.path.join(path, file_))
    return paths_histories


def plot_model_loss_comparison(model_histories: List[dict]) -> List[plt.Figure]:
    fig, axes = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
    x_min = 0
    x_max = 0
    for history in sorted(model_histories, key=lambda x: x['model_name']):
        if 'loss' not in history or 'val_loss' not in history:
            continue
        axes[0].plot(history['loss'], label=history['model_name'])
        axes[1].plot(history['val_loss'])
        x_max = max(x_max, *history['loss'], *history['val_loss'])
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
    fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.0, -0.1, 1.0, 1.0), bbox_transform=plt.gcf().transFigure)
    axes[0].set_title('Training loss')
    axes[1].set_title('Validation loss')
    return [fig]


def plot_model_timing_comparison(model_histories: List[dict]) -> List[plt.Figure]:
    # TODO:  add validation/test timings
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list()
    timings = list()
    for history in sorted(model_histories, key=lambda x: x['model_name']):
        if 'train_start' not in history or 'train_finish' not in history:
            continue
        labels.append(history['model_name'])
        timings.append((history['train_finish'] - history['train_start']).seconds / 60)
    ax.barh(np.arange(len(timings)), timings, tick_label=labels)
    ax.set_xlabel('Minutes')
    ax.set_title('Training times')
    return [fig]
