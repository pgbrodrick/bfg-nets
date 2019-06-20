from typing import List

import matplotlib.pyplot as plt

from rsCNN.evaluation import samples
from rsCNN.evaluation.visualizations import figures as viz_figures, subplots


def plot_data_input_samples(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_samples_per_page: int = 10,
        max_features_per_page: int = 5,
        max_responses_per_page: int = 5
) -> List[plt.Figure]:
    figures = viz_figures.plot_figures_iterating_through_samples_features_responses(
        sampled, _plot_inputs_page, max_pages, max_samples_per_page, max_features_per_page, max_responses_per_page
    )
    for idx, figure in enumerate(figures):
        figure.suptitle('{} Sequence Input Samples (page {})'.format(sampled.data_sequence_label or '', idx + 1))
    return figures


def _plot_inputs_page(
        sampled: samples.Samples,
        range_samples: range,
        range_features: range,
        range_responses: range
) -> plt.Figure:
    nrows = len(range_samples)
    num_features_plots = 2 * len(range_features)
    num_responses_plots = 2 * len(range_responses)
    num_weights_plots = 1
    ncols = num_features_plots + num_responses_plots + num_weights_plots
    fig, grid = viz_figures.get_figure_and_grid(nrows, ncols)
    for idx_sample in range_samples:
        axes = viz_figures.get_axis_iterator_for_sample_row(grid, idx_sample)
        for idx_feature in range_features:
            subplots.plot_raw_features(
                sampled, idx_sample, idx_feature, axes.__next__(), idx_sample == 0, idx_feature == 0)
        for idx_feature in range_features:
            subplots.plot_transformed_features(
                sampled, idx_sample, idx_feature, axes.__next__(), idx_sample == 0, False)
        for idx_response in range_responses:
            subplots.plot_raw_responses(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
        for idx_response in range_responses:
            subplots.plot_transformed_responses(
                sampled, idx_sample, idx_response, axes.__next__(), idx_sample == 0, False)
        subplots.plot_weights(sampled, idx_sample, axes.__next__(), idx_sample == 0, False)
    return fig
