from typing import List

import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import samples, shared


# TODO:  I want to see one-hot encoded categories, e.g., both geomorphic and benthic, as single categorical plots

def plot_raw_and_transformed_result_examples(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_samples_per_page: int = 10,
        max_features_per_page: int = 5,
        max_responses_per_page: int = 5
) -> List[plt.Figure]:
    # TODO:  allow user to configure which features, if any, show on results plot (currently none)
    figures = shared.plot_figures_iterating_through_samples_features_responses(
        sampled, _plot_results_page, max_pages, max_samples_per_page, max_features_per_page, max_responses_per_page
    )
    for idx, figure in enumerate(figures):
        figure.suptitle('Prediction Example Plots (page {})'.format(idx))
    return figures


def _plot_results_page(
        sampled: samples.Samples,
        range_samples: range,
        range_responses: range
) -> plt.Figure:
    nrows = len(range_samples)
    ncols = 1 + 4 * len(range_responses)
    fig, grid = shared.get_figure_and_grid(nrows, ncols)
    for idx_sample in range_samples:
        idx_col = 0
        for idx_response in range_responses:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_raw_responses(sampled, idx_sample, idx_response, ax, idx_sample == 0, idx_col == 0)
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_transformed_responses(sampled, idx_sample, idx_response, ax, idx_sample == 0, False)
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_raw_predictions(sampled, idx_sample, idx_response, ax, idx_sample == 0, False)
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_transformed_predictions(sampled, idx_sample, idx_response, ax, idx_sample == 0, False)
        ax = plt.subplot(grid[idx_sample, idx_col])
        shared.plot_weights(sampled, ax, idx_sample == 0)
    return fig


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


def spatial_error(
        model: keras.Model,
        data_sequence: BaseSequence
):

    # TODO: Consider handling weights
    fig_list = []

    # TODO: deal with more than one batch....
    features, responses = data_sequence.__getitem__(0)
    pred_responses = model.predict(features)
    features = features[0]
    responses = responses[0]
    responses, weights = responses[..., :-1], responses[..., -1]

    diff = np.abs(responses-pred_responses)

    max_resp_per_page = min(8, responses.shape[-1])

    _response_ind = 0
    while (_response_ind < responses.shape[-1]):
        fig = plt.figure(figsize=(25, 8))
        gs1 = gridspec.GridSpec(1, max_resp_per_page)
        for _r in range(_response_ind, min(responses.shape[-1], _response_ind + max_resp_per_page)):
            ax = plt.subplot(gs1[0, _r - _response_ind])

            vset = diff.copy()
            vset[weights == 0] = np.nan
            vset = np.nanmean(vset, axis=0)
            ax.imshow(np.nanmean(diff[..., _r], axis=0), vmin=np.nanmin(
                vset), vmax=np.nanmax(vset))
            plt.title('Response ' + str(_r))
            plt.axis('off')

        _response_ind += max_resp_per_page
        plt.suptitle('Response Spatial Deviation ' + str((len(fig_list))))
        fig_list.append(fig)

    return fig_list
