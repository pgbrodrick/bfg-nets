from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from rsCNN.reporting import samples
from rsCNN.reporting.visualizations import figures as viz_figures, subplots


def plot_model_output_samples(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_samples_per_page: int = 10,
        max_features_per_page: int = 5,
        max_responses_per_page: int = 5
) -> List[plt.Figure]:
    figures = viz_figures.plot_figures_iterating_through_samples_features_responses(
        sampled, _plot_output_page, max_pages, max_samples_per_page, max_features_per_page, max_responses_per_page
    )
    for idx, figure in enumerate(figures):
        figure.suptitle('{} Sequence Prediction Samples (page {})'.format(sampled.data_sequence_label or '', idx + 1))
    return figures


def _plot_output_page(
        sampled: samples.Samples,
        range_samples: range,
        range_features: range,
        range_responses: range
) -> plt.Figure:
    nrows = len(range_samples)
    has_softmax = sampled.config.architecture.output_activation == 'softmax'
    num_responses_plots = 2 * len(range_responses)
    num_predictions_plots = num_responses_plots
    num_regression_plots = 2 * len(range_responses) * int(not has_softmax)
    num_categorical_plots = 2 * int(has_softmax)
    num_weights_plots = 1
    ncols = (num_responses_plots + num_predictions_plots + num_regression_plots + num_categorical_plots +
             num_weights_plots)
    fig, grid = viz_figures.get_figure_and_grid(nrows, ncols)
    for idx_sample in range_samples:
        axes = viz_figures.get_axis_iterator_for_sample_row(grid, idx_sample)
        for idx_response in range_responses:
            subplots.plot_raw_responses(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, idx_response == 0)
            subplots.plot_transformed_responses(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
            subplots.plot_raw_predictions(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
            subplots.plot_transformed_predictions(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
            if not has_softmax:
                subplots.plot_raw_error_regression(
                    sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
                subplots.plot_transformed_error_regression(
                    sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
        # Note: if softmax aka categorical, then we only need one plot per sample, not per response
        if has_softmax:
            subplots.plot_max_likelihood(sampled, idx_sample, axes.__next__(), idx_sample == 0, False)
            subplots.plot_binary_error(sampled, idx_sample, axes.__next__(), idx_sample == 0, False)
        subplots.plot_weights(sampled, idx_sample, axes.__next__(), idx_sample == 0, False)
    return fig


def plot_single_sequence_prediction_histogram(
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
