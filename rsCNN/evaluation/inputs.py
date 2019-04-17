from typing import List

import matplotlib.pyplot as plt

from rsCNN.evaluation import samples, shared


# TODO:  I want to see one-hot encoded geomorphic categories together in input plots

def plot_raw_and_transformed_input_samples(
        sampled: samples.Samples,
        max_pages: int = 8,
        max_samples_per_page: int = 10,
        max_features_per_page: int = 5,
        max_responses_per_page: int = 5
) -> List[plt.Figure]:
    figures = shared.plot_figures_iterating_through_samples_features_responses(
        sampled, _plot_inputs_page, max_pages, max_samples_per_page, max_features_per_page, max_responses_per_page
    )
    for idx, figure in enumerate(figures):
        figure.suptitle('Input Example Plots (page {})'.format(idx))
    return figures


def _plot_inputs_page(
        sampled: samples.Samples,
        range_samples: range,
        range_features: range,
        range_responses: range
) -> plt.Figure:
    nrows = len(range_samples)
    ncols = 1 + 2 * (len(range_features) + len(range_responses))
    fig, grid = shared.get_figure_and_grid(nrows, ncols)
    for idx_sample in range_samples:
        idx_col = 0
        for idx_feature in range_features:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_raw_features(sampled, idx_sample, idx_feature, ax, idx_sample == 0, idx_col == 0)
            idx_col += 1
        for idx_feature in range_features:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_transformed_features(sampled, idx_sample, idx_feature, ax, idx_sample == 0, False)
            idx_col += 1
        for idx_response in range_responses:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_raw_responses(sampled, idx_sample, idx_response, ax, idx_sample == 0, False)
            idx_col += 1
        for idx_response in range_responses:
            ax = plt.subplot(grid[idx_sample, idx_col])
            shared.plot_transformed_responses(sampled, idx_sample, idx_response, ax, idx_sample == 0, False)
            idx_col += 1
        ax = plt.subplot(grid[idx_sample, idx_col])
        shared.plot_weights(sampled, ax, idx_sample == 0)
    return fig
