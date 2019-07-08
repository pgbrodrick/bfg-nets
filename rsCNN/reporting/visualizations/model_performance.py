import logging
from typing import List

from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sklearn.metrics

from rsCNN.reporting import samples
from rsCNN.reporting.visualizations import colormaps, subplots


_logger = logging.getLogger(__name__)


def plot_classification_report(sampled: samples.Samples) -> List[plt.Figure]:
    if not sampled.raw_responses or not sampled.raw_predictions:
        _logger.debug('Confusion matrix not plotted; no responses or predictions available.')
        return list()

    classes, actual, predicted = _calculate_classification_classes_actual_and_predicted(sampled)
    report = 'Classification report\n\n' + sklearn.metrics.classification_report(actual, predicted, classes)
    fig, ax = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
    ax.text(0, 0, report, **{'fontsize': 8, 'fontfamily': 'monospace'})
    ax.axis('off')
    fig.suptitle('{} Sequence Classification Report'.format(sampled.data_sequence_label))
    return [fig]


def plot_confusion_matrix(sampled: samples.Samples) -> [plt.Figure]:
    if not sampled.raw_responses or not sampled.raw_predictions:
        _logger.debug('Confusion matrix not plotted; no responses or predictions available.')
        return list()

    classes, actual, predicted = _calculate_classification_classes_actual_and_predicted(sampled)
    confusion_matrix = sklearn.metrics.confusion_matrix(actual, predicted, labels=classes)
    normed_matrix = confusion_matrix.astype(float) / confusion_matrix.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(figsize=(16, 8), nrows=1, ncols=2)

    for idx_ax, ax in enumerate(axes):
        if idx_ax == 0:
            title = 'Confusion matrix, with counts'
            matrix = confusion_matrix
            value_format = 'd'
            max_ = np.nanmax(matrix)
        elif idx_ax == 1:
            title = 'Normalized confusion matrix'
            matrix = normed_matrix
            value_format = '.2f'
            max_ = 1
        im = ax.imshow(matrix, interpolation='nearest', vmin=0, vmax=max_, cmap=colormaps.COLORMAP_METRICS)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]), xticklabels=classes,
               yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')

        # Matrix element labels
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], value_format), ha='center', va='center',
                        color='white' if matrix[i, j] > max_ / 2. else 'black')

    fig.suptitle('{} Sequence Confusion Matrix'.format(sampled.data_sequence_label or ''))
    plt.tight_layout(h_pad=1)
    return [fig]


def _calculate_classification_classes_actual_and_predicted(sampled):
    classes = range(sampled.num_responses)
    actual = np.argmax(sampled.raw_responses, axis=-1).ravel()
    actual = actual[np.isfinite(actual)]
    predicted = np.argmax(sampled.raw_predictions, axis=-1).ravel()
    predicted = predicted[np.isfinite(predicted)]
    return classes, actual, predicted


def plot_spatial_classification_error(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_responses_per_page: int = 10
) -> List[plt.Figure]:
    actual = np.expand_dims(np.argmax(sampled.raw_responses, axis=-1), -1)
    predicted = np.expand_dims(np.argmax(sampled.raw_predictions, axis=-1), -1)
    error = (actual != predicted).astype(float)
    is_finite = np.logical_and(
        np.isfinite(sampled.raw_responses).all(axis=-1),
        np.isfinite(sampled.raw_predictions).all(axis=-1)
    )
    error[~is_finite] = np.nan
    error = np.nanmean(error, axis=0)
    return _plot_spatial_error(error, sampled, max_pages, max_responses_per_page)


def plot_spatial_regression_error(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_responses_per_page: int = 10,
) -> List[plt.Figure]:
    abs_error = np.nanmean(np.abs(sampled.raw_predictions - sampled.raw_responses), axis=0)
    return _plot_spatial_error(abs_error, sampled, max_pages, max_responses_per_page)


def _plot_spatial_error(
        error: np.array,
        sampled: samples.Samples,
        max_pages: int,
        max_responses_per_page: int,
) -> List[plt.Figure]:
    figures = []

    num_pages = min(max_pages, np.ceil(sampled.num_responses / max_responses_per_page))

    loss_window_radius = sampled.config.data_build.loss_window_radius
    window_radius = sampled.config.data_build.window_radius
    buffer = int(window_radius - loss_window_radius)

    def _get_axis_generator_for_page(grid, num_rows, num_cols):
        for idx_col in range(num_cols):
            for idx_row in range(num_rows):
                yield plt.subplot(grid[idx_row, idx_col])

    idx_page = 0
    idx_response = 0
    while idx_page < num_pages:
        num_figs_on_page = min(max_responses_per_page, error.shape[-1] - idx_response)
        nrows = 1
        ncols = num_figs_on_page
        width = 30 * ncols / (nrows + ncols)
        height = 30 * nrows / (nrows + ncols)
        fig = plt.figure(figsize=(width, height))
        grid = gridspec.GridSpec(nrows, ncols)
        for ax in _get_axis_generator_for_page(grid, nrows, ncols):
            min_ = 0
            max_ = np.nanmax(error[buffer:-buffer, buffer:-buffer, idx_response])
            ax.imshow(error[..., idx_response], vmin=min_, vmax=max_, cmap=colormaps.COLORMAP_ERROR)
            ax.set_xlabel('Response {}'.format(idx_response))
            ax.xaxis.set_label_position('top')
            ax.set_xticks([])
            ax.set_yticks([])
            subplots.add_internal_window_to_subplot(sampled, ax)
            idx_response += 1
            if idx_response > error.shape[-1]:
                break
        figures.append(fig)
        idx_page += 1
        fig.suptitle('{} Sequence Response Spatial Deviation (page {})'.format(
            sampled.data_sequence_label or '', idx_page + 1))
    return figures
