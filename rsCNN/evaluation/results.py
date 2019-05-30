from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sklearn.metrics

from rsCNN.evaluation import samples, shared


# TODO:  I want to see one-hot encoded categories, e.g., both geomorphic and benthic, as single categorical plots


def print_classification_report(sampled: samples.Samples) -> List[plt.Figure]:
    classes, actual, predicted = _calculate_classification_classes_actual_and_predicted(sampled)
    report = 'Classification report\n\n' + sklearn.metrics.classification_report(actual, predicted, classes)
    fig, ax = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
    ax.text(0, 0, report, **{'fontsize': 8, 'fontfamily': 'monospace'})
    ax.axis('off')
    fig.suptitle('{} Sequence Classification Report'.format(sampled.data_sequence_label))
    return [fig]


def plot_confusion_matrix(sampled: samples.Samples) -> [plt.Figure]:
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
        else:
            title = 'Normalized confusion matrix'
            matrix = normed_matrix
            value_format = '.2f'
            max_ = 1
        im = ax.imshow(matrix, interpolation='nearest', vmin=0, vmax=max_, cmap=shared.COLORMAP_METRICS)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(im, ax=cax, )
        ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]), xticklabels=classes,
               yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')
        # Matrix element labels
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], value_format), ha='center', va='center',
                        color='white' if matrix[i, j] > max_ / 2. else 'black')
    fig.suptitle('{} Sequence Confusion Matrix'.format(sampled.data_sequence_label or ''))
    return [fig]


def _calculate_classification_classes_actual_and_predicted(sampled):
    classes = range(sampled.num_responses)
    actual = np.argmax(sampled.raw_responses, axis=-1).ravel()
    actual = actual[np.isfinite(actual)]
    predicted = np.argmax(sampled.raw_predictions, axis=-1).ravel()
    predicted = predicted[np.isfinite(predicted)]
    return classes, actual, predicted


def plot_raw_and_transformed_prediction_samples(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_samples_per_page: int = 10,
        max_features_per_page: int = 5,
        max_responses_per_page: int = 5
) -> List[plt.Figure]:
    # TODO:  allow user to configure which features, if any, show on results plot (currently none)
    figures = shared.plot_figures_iterating_through_samples_features_responses(
        sampled, _plot_predictions_page, max_pages, max_samples_per_page, max_features_per_page, max_responses_per_page
    )
    for idx, figure in enumerate(figures):
        figure.suptitle('{} Sequence Prediction Samples (page {})'.format(sampled.data_sequence_label or '', idx + 1))
    return figures


def _plot_predictions_page(
        sampled: samples.Samples,
        range_samples: range,
        range_features: range,
        range_responses: range
) -> plt.Figure:
    # TODO:  the has_softmax check needs to be updated in the future. I think it's probably still useful to show the
    #  most likely class predicted by the softmax, but it's not a sufficient check for categorical data. That means
    #  that the regression or categorical plot check below is going to be wrong in some cases. We can wait to see how
    #  Phil handles data types in the data config.
    nrows = len(range_samples)
    has_softmax = sampled.config.architecture.output_activation == 'softmax'
    num_responses_plots = 2 * len(range_responses)
    num_predictions_plots = num_responses_plots
    num_regression_plots = 2 * len(range_responses) * int(not has_softmax)
    num_categorical_plots = 2 * len(range_responses) * int(has_softmax)
    num_weights_plots = 1
    ncols = (num_responses_plots + num_predictions_plots + num_regression_plots + num_categorical_plots +
             num_weights_plots)
    fig, grid = shared.get_figure_and_grid(nrows, ncols)
    for idx_sample in range_samples:
        axes = shared.get_axis_iterator_for_sample_row(grid, idx_sample)
        for idx_response in range_responses:
            shared.plot_raw_responses(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, idx_response == 0)
            shared.plot_transformed_responses(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
            shared.plot_raw_predictions(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
            shared.plot_transformed_predictions(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
            if not has_softmax:
                shared.plot_raw_error_regression(
                    sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
                shared.plot_transformed_error_regression(
                    sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
        # Note: if softmax aka categorical, then we only need one plot per sample, not per response
        if has_softmax:
            # TODO:  we're not really "plotting softmax", what's a better name for this? It's escaping me!
            shared.plot_softmax(sampled, idx_sample, axes.__next__(), idx_sample == 0, False)
            # TODO:  same naming issue here
            shared.plot_error_categorical(sampled, idx_sample, axes.__next__(), idx_sample == 0, False)
        shared.plot_weights(sampled, idx_sample, axes.__next__(), idx_sample == 0, False)
    return fig


def single_sequence_prediction_histogram(
        sampled: samples.Samples,
        max_responses_per_page: int = 15
):
    # TODO:  there is an indexing error in this function when you have more responses than fit on a single page
    max_responses_per_page = min(max_responses_per_page, sampled.num_responses)
    _response_ind = 0

    # Training Raw Space
    fig_list = []
    while _response_ind < sampled.num_responses:

        fig = plt.figure(figsize=(6 * max_responses_per_page, 10))
        gs1 = gridspec.GridSpec(4, max_responses_per_page)
        for _r in range(_response_ind, min(_response_ind+max_responses_per_page, sampled.num_responses)):
            ax = plt.subplot(gs1[0, _r])
            b, h = _get_lhist(sampled.trans_responses[..., _r])
            ax.plot(h, b, color='black')
            b, h = _get_lhist(sampled.trans_predictions[..., _r])
            ax.plot(h, b, color='green')

            if (_r == _response_ind):
                plt.ylabel('Raw')
            plt.title('Response ' + str(_r))

            ax = plt.subplot(gs1[1, _r])

            b, h = _get_lhist(sampled.raw_responses[..., _r])
            ax.plot(h, b, color='black')
            b, h = _get_lhist(sampled.raw_predictions[..., _r])
            ax.plot(h, b, color='green')

            if (_r == _response_ind):
                plt.ylabel('Transformed')

        _response_ind += max_responses_per_page
        fig_list.append(fig)
        fig.suptitle('{} Sequence Response Histogram (page {})'.format(sampled.data_sequence_label, len(fig_list)))
    return fig_list


def _get_lhist(data, bins=10):
    hist, edge = np.histogram(data, bins=bins, range=(np.nanmin(data), np.nanmax(data)))
    hist = hist.tolist()
    edge = edge.tolist()
    phist = [0]
    pedge = [edge[0]]
    for _e in range(0, len(edge)-1):
        phist.append(hist[_e])
        phist.append(hist[_e])

        pedge.append(edge[_e])
        pedge.append(edge[_e+1])

    phist.append(0)
    pedge.append(edge[-1])
    phist = np.array(phist)
    pedge = np.array(pedge)
    return phist, pedge


def plot_spatial_categorical_error(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_responses_per_row: int = 10,
        max_rows_per_page: int = 10
) -> List[plt.Figure]:
    # TODO: Consider handling weights, already handle 0 weights but could weight errors differently
    # TODO: This assumes that categorical variables are one-hot encoded
    actual = np.expand_dims(np.argmax(sampled.raw_responses, axis=-1), -1)
    predicted = np.expand_dims(np.argmax(sampled.raw_predictions, axis=-1), -1)
    error = (actual != predicted).astype(float)
    is_finite = np.logical_and(
        np.isfinite(sampled.raw_responses).all(axis=-1),
        np.isfinite(sampled.raw_predictions).all(axis=-1)
    )
    error[~is_finite] = np.nan
    error = np.nanmean(error, axis=0)
    return _plot_spatial_error(error, sampled, max_pages, max_responses_per_row, max_rows_per_page)


def plot_spatial_regression_error(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_responses_per_row: int = 10,
        max_rows_per_page: int = 10
) -> List[plt.Figure]:
    # TODO: Consider handling weights, already handle 0 weights but could weight errors differently
    abs_error = np.nanmean(np.abs(sampled.raw_predictions - sampled.raw_responses), axis=0)
    return _plot_spatial_error(abs_error, sampled, max_pages, max_responses_per_row, max_rows_per_page)


def _plot_spatial_error(
    error: np.array,
    sampled: samples.Samples,
    max_pages: int,
    max_responses_per_row: int,
    max_rows_per_page: int
) -> List[plt.Figure]:
    figures = []

    max_responses_per_page = max_responses_per_row * max_rows_per_page
    num_pages = min(max_pages, np.ceil(sampled.num_responses / max_responses_per_page))

    inshape = sampled.config.architecture.inshape
    loss_window_radius = sampled.config.data_build.loss_window_radius
    buffer = int((inshape[0] - loss_window_radius * 2) / 2)

    def _get_axis_generator_for_page(grid, num_rows, num_cols):
        for idx_col in range(num_cols):
            for idx_row in range(num_rows):
                yield plt.subplot(grid[idx_row, idx_col])

    idx_page = 0
    idx_response = 0
    while idx_page < num_pages:
        num_figs_on_page = min(max_responses_per_page, error.shape[-1] - idx_response)
        nrows = int(np.ceil(num_figs_on_page / max_responses_per_row))
        ncols = int(min(max_responses_per_row, num_figs_on_page))
        fig, grid = shared.get_figure_and_grid(nrows, ncols)
        for ax in _get_axis_generator_for_page(grid, nrows, ncols):
            min_ = 0
            max_ = np.nanmax(error[buffer:-buffer, buffer:-buffer, idx_response])
            ax.imshow(error[..., idx_response], vmin=min_, vmax=max_, cmap=shared.COLORMAP_ERROR)
            ax.set_xlabel('Response {}'.format(idx_response))
            ax.xaxis.set_label_position('top')
            ax.set_xticks([])
            ax.set_yticks([])
            shared.add_internal_window_to_subplot(sampled, ax)
            idx_response += 1
            if idx_response > error.shape[-1]:
                break
        figures.append(fig)
        idx_page += 1
        fig.suptitle('{} Sequence Response Spatial Deviation (page {})'.format(
            sampled.data_sequence_label or '', idx_page + 1))
    return figures
