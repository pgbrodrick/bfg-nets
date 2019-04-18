from typing import List

import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import samples, shared


# TODO:  I want to see one-hot encoded categories, e.g., both geomorphic and benthic, as single categorical plots

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
        figure.suptitle('Prediction Example Plots (page {})'.format(idx))
    return figures


def _plot_predictions_page(
        sampled: samples.Samples,
        range_samples: range,
        range_responses: range
) -> plt.Figure:
    # TODO:  the has_softmax check needs to be updated in the future. I think it's probably still useful to show the
    #  most likely class predicted by the softmax, but it's not a sufficient check for categorical data. That means
    #  that the regression or categorical plot check below is going to be wrong in some cases. We can wait to see how
    #  Phil handles data types in the data config.
    has_softmax = sampled.network_config['architecture_options']['output_activation'] == 'softmax'
    nrows = len(range_samples)
    ncols = 1 + (4 + 2 * int(not has_softmax)) * len(range_responses) + 2 * int(has_softmax)
    fig, grid = shared.get_figure_and_grid(nrows, ncols)
    for idx_sample in range_samples:
        axes = shared.get_axis_iterator_for_sample_row(grid, idx_sample)
        for idx_response in range_responses:
            shared.plot_raw_responses(
                sampled, idx_sample, idx_response, axes.next(), idx_sample == 0, idx_response == 0)
            shared.plot_transformed_responses(
                sampled, idx_sample, idx_response, axes.next(), idx_sample == 0, False)
            shared.plot_raw_predictions(
                sampled, idx_sample, idx_response, axes.next(), idx_sample == 0, False)
            shared.plot_transformed_predictions(
                sampled, idx_sample, idx_response, axes.next(), idx_sample == 0, False)
            if not has_softmax:
                shared.plot_raw_error_regression(
                    sampled, idx_sample, idx_response, axes.next(), idx_sample == 0, False)
                shared.plot_transformed_error_regression(
                    sampled, idx_sample, idx_response, axes.next(), idx_sample == 0, False)
        # Note: if softmax aka categorical, then we only need one plot per sample, not per response
        if has_softmax:
            # TODO:  we're not really "plotting softmax", what's a better name for this? It's escaping me!
            shared.plot_softmax(sampled, idx_sample, axes.next(), idx_sample == 0, False)
            # TODO:  same naming issue here
            shared.plot_error_categorical(sampled, idx_sample, axes.next(), idx_sample == 0, False)
        shared.plot_weights(sampled, idx_sample, axes.next(), idx_sample == 0, False)
    return fig


def single_sequence_prediction_histogram(
        model: keras.Model,
        data_sequence: BaseSequence,
        seq_str: str = ''
):

        # TODO: deal with more than one batch....
    features, responses = data_sequence.__getitem__(0)
    pred_responses = model.predict(features)
    features = features[0]
    responses = responses[0]
    responses, weights = responses[..., :-1], responses[..., -1]

    responses[weights == 0, :] = np.nan
    pred_responses[weights == 0, :] = np.nan

    invtrans_responses = data_sequence.response_scaler.inverse_transform(responses)
    invtrans_pred_responses = data_sequence.response_scaler.inverse_transform(pred_responses)

    responses = responses.reshape((-1, responses.shape[-1]))
    pred_responses = pred_responses.reshape((-1, pred_responses.shape[-1]))
    invtrans_responses = invtrans_responses.reshape((-1, invtrans_responses.shape[-1]))
    invtrans_pred_responses = invtrans_pred_responses.reshape((-1, invtrans_pred_responses.shape[-1]))

    max_resp_per_page = min(8, responses.shape[-1])
    _response_ind = 0

    # Training Raw Space
    fig_list = []
    while _response_ind < responses.shape[-1]:

        fig = plt.figure(figsize=(6 * max_resp_per_page, 10))
        gs1 = gridspec.GridSpec(4, max_resp_per_page)
        for _r in range(_response_ind, min(_response_ind+max_resp_per_page, responses.shape[-1])):
            ax = plt.subplot(gs1[0, _r])
            b, h = _get_lhist(responses[..., _r])
            plt.plot(h, b, color='black')
            b, h = _get_lhist(pred_responses[..., _r])
            plt.plot(h, b, color='green')

            if (_r == _response_ind):
                plt.ylabel('Raw')
            plt.title('Response ' + str(_r))

            ax = plt.subplot(gs1[1, _r])

            b, h = _get_lhist(invtrans_responses[..., _r])
            plt.plot(h, b, color='black')
            b, h = _get_lhist(invtrans_pred_responses[..., _r])
            plt.plot(h, b, color='green')

            if (_r == _response_ind):
                plt.ylabel('Transformed')

        _response_ind += max_resp_per_page
        plt.suptitle(seq_str + ' Response Histogram Page ' + str((len(fig_list))))
        fig_list.append(fig)
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


def plot_spatial_regression_error(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_responses_per_row: int = 10,
        max_rows_per_page: int = 10
) -> List[plt.Figure]:
    # TODO: Consider handling weights, already handle 0 weights but could weight errors differently
    abs_error = np.nanmean(np.abs(sampled.raw_predictions, - sampled.raw_responses), axis=0)

    figures = []
    num_pages = min(max_pages, np.ceil(sampled.num_responses / (max_responses_per_row * max_rows_per_page)))

    def _get_axis_generator_for_page(grid, num_rows, num_cols):
        for idx_col in range(num_cols):
            for idx_row in range(num_rows):
                yield plt.subplot(grid[idx_row, idx_col])

    idx_page = 0
    idx_response = 0
    while idx_page < num_pages and idx_response < sampled.num_responses:
        fig, grid = shared.get_figure_and_grid(max_rows_per_page, max_responses_per_row)
        for ax in _get_axis_generator_for_page(grid, max_rows_per_page, max_responses_per_row):
            min_ = np.nanmin(abs_error[:, :, idx_response][sampled.weights != 0])
            max_ = np.nanmax(abs_error[:, :, idx_response][sampled.weights != 0])
            ax.imshow(abs_error[:, idx_response], vmin=min_, vmax=max_, cmap=shared.COLORMAP_ERROR)
            ax.set_xlabel('Response {}'.format(idx_response))
            ax.xaxis.set_label_position('top')
            ax.set_xticks([])
            ax.set_yticks([])
            idx_response += 1
        fig.suptitle('Response Spatial Deviation {}'.format(idx_page))
        figures.append(fig)
        idx_page += 1
    return figures
