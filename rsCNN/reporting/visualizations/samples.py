from typing import Iterator, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from rsCNN.reporting import samples
from rsCNN.reporting.visualizations import figures as viz_figures, subplots


LABEL_CLASSIFICATION = 'CLASSIFICATION'
LABEL_REGRESSION = 'REGRESSION'


def plot_classification_samples(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_samples_per_page: int = 10,
        max_features_per_page: int = 5,
        max_responses_per_page: int = 5
) -> List[plt.Figure]:
    return _plot_samples(
        sampled, max_pages, max_samples_per_page, max_features_per_page, max_responses_per_page,
        sample_type=LABEL_CLASSIFICATION
    )


def plot_regression_samples(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_samples_per_page: int = 10,
        max_features_per_page: int = 5,
        max_responses_per_page: int = 5
) -> List[plt.Figure]:
    return _plot_samples(
        sampled, max_pages, max_samples_per_page, max_features_per_page, max_responses_per_page,
        sample_type=LABEL_REGRESSION
    )


def _plot_samples(
    sampled: samples.Samples,
    max_pages: int,
    max_samples_per_page: int,
    max_features_per_page: int,
    max_responses_per_page: int,
    sample_type: str
) -> List[plt.Figure]:
    # Calculate figure parameters
    figures = list()
    num_pages = min(max_pages, np.ceil(sampled.num_samples / max_samples_per_page))
    num_features = min(max_features_per_page, sampled.num_features)
    num_responses = min(max_responses_per_page, sampled.num_responses)
    if sample_type is LABEL_CLASSIFICATION:
        num_subplots = _calculate_number_classification_subplots(num_features)
        sample_plotter = _plot_classification_sample
    elif sample_type is LABEL_REGRESSION:
        num_subplots = _calculate_number_regression_subplots(num_features, num_responses)
        sample_plotter = _plot_regression_sample

    # Iterate through pages and samples
    for idx_page in range(num_pages):
        fig, grid = viz_figures.get_figure_and_grid(max_samples_per_page, num_subplots)
        idxs_samples = range(idx_page * max_samples_per_page, (1 + idx_page) * max_samples_per_page)
        for idx_sample in idxs_samples:
            sample_axes = iter([plt.subplot(grid[idx_sample, idx_subplot]) for idx_subplot in range(num_subplots)])
            sample_plotter(sampled, idx_sample, num_features, num_responses, sample_axes)
        fig.suptitle('{}Sequence Samples (page {})'.format(sampled.data_sequence_label + ' ' or '', idx_page + 1))
        figures.append(fig)

    return figures


def _calculate_number_classification_subplots(num_features: int) -> int:
    num_feature_plots = 2 * num_features  # x2 for the raw and transformed features
    num_response_plots = 1  # All responses shown simultaneously
    num_prediction_plots = 2  # 2 for the raw and max likelihood predictions
    num_error_plots = 1
    num_weights_plots = 1
    return num_feature_plots + num_response_plots + num_prediction_plots + num_error_plots + num_weights_plots


def _calculate_number_regression_subplots(num_features: int, num_responses: int) -> int:
    num_feature_plots = 2 * num_features  # x2 for the raw and transformed features
    num_response_plots = 2 * num_responses  # x2 for the raw and transformed responses
    num_prediction_plots = 2 * num_responses  # x2 for the raw and transformed responses
    num_error_plots = 1
    num_weights_plots = 1
    return num_feature_plots + num_response_plots + num_prediction_plots + num_error_plots + num_weights_plots


def _plot_classification_sample(
        sampled: samples.Samples,
        idx_sample: int,
        num_features: int,
        num_responses: int,
        sample_axes: Iterator
) -> None:
    for idx_feature in range(num_features):
        subplots.plot_raw_features(
            sampled, idx_sample, idx_feature, sample_axes.__next__(), idx_sample == 0, idx_feature == 0)

    for idx_feature in range(num_features):
        subplots.plot_transformed_features(
            sampled, idx_sample, idx_feature, sample_axes.__next__(), idx_sample == 0, False)

    subplots.plot_categorical_responses()

    subplots.plot_max_likelihood_classification(
        sampled, idx_sample, sample_axes.__next__(), idx_sample == 0, False)

    subplots.plot_binary_error_classification(sampled, idx_sample, sample_axes.__next__(), idx_sample == 0, False)

    for idx_response in range(num_responses):
        subplots.plot_raw_predictions(
            sampled, idx_sample, idx_response, sample_axes.__next__(), idx_sample == 0, False)

    subplots.plot_weights(sampled, idx_sample, sample_axes.__next__(), idx_sample == 0, False)


def _plot_regression_sample(
        sampled: samples.Samples,
        idx_sample: int,
        num_features: int,
        num_responses: int,
        sample_axes: Iterator
) -> None:
    for idx_feature in range(num_features):
        subplots.plot_raw_features(
            sampled, idx_sample, idx_feature, sample_axes.__next__(), idx_sample == 0, idx_feature == 0)

    for idx_feature in range(num_features):
        subplots.plot_transformed_features(
            sampled, idx_sample, idx_feature, sample_axes.__next__(), idx_sample == 0, False)

    for idx_response in range(num_responses):
        subplots.plot_raw_responses(
            sampled, idx_sample, idx_response, sample_axes.__next__(), idx_sample == 0, False)

    for idx_response in range(num_responses):
        subplots.plot_transformed_responses(
            sampled, idx_sample, idx_response, sample_axes.__next__(), idx_sample == 0, False)

    for idx_response in range(num_responses):
        subplots.plot_raw_predictions(
            sampled, idx_sample, idx_response, sample_axes.__next__(), idx_sample == 0, False)

    for idx_response in range(num_responses):
        subplots.plot_transformed_responses(
            sampled, idx_sample, idx_response, sample_axes.__next__(), idx_sample == 0, False)

    subplots.plot_raw_error_regression(
        sampled, idx_sample, idx_response, sample_axes.__next__(), idx_sample == 0, False)

    subplots.plot_transformed_error_regression(
        sampled, idx_sample, idx_response, sample_axes.__next__(), idx_sample == 0, False)

    subplots.plot_weights(sampled, idx_sample, sample_axes.__next__(), idx_sample == 0, False)


def plot_single_sequence_prediction_histogram(
        sampled: samples.Samples,
        max_responses_per_page: int = 15
):
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
