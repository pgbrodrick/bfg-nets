from typing import Union

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

from rsCNN.evaluation import samples
from rsCNN.evaluation.shared import colormaps


plt.switch_backend('Agg')  # Needed for remote server plotting


_FLOAT_DECIMALS = 2


def plot_raw_features(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    _plot_sample_attribute(sampled, idx_sample, idx_feature, 'raw_features', ax, add_xlabel, add_ylabel)


def plot_transformed_features(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    _plot_sample_attribute(sampled, idx_sample, idx_feature, 'trans_features', ax, add_xlabel, add_ylabel)


def plot_raw_responses(
        sampled: samples.Samples,
        idx_sample: int,
        idx_response: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    _plot_sample_attribute(sampled, idx_sample, idx_response, 'raw_responses', ax, add_xlabel, add_ylabel)


def plot_transformed_responses(
        sampled: samples.Samples,
        idx_sample: int,
        idx_response: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    _plot_sample_attribute(sampled, idx_sample, idx_response, 'trans_responses', ax, add_xlabel, add_ylabel)


def plot_raw_predictions(
        sampled: samples.Samples,
        idx_sample: int,
        idx_response: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    _plot_sample_attribute(sampled, idx_sample, idx_response, 'raw_predictions', ax, add_xlabel, add_ylabel)
    add_internal_window_to_subplot(sampled, ax)


def plot_transformed_predictions(
        sampled: samples.Samples,
        idx_sample: int,
        idx_response: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    _plot_sample_attribute(sampled, idx_sample, idx_response, 'trans_predictions', ax, add_xlabel, add_ylabel)
    add_internal_window_to_subplot(sampled, ax)


def _plot_sample_attribute(
        sampled: samples.Samples,
        idx_sample: int,
        idx_axis: int,
        attribute_name: str,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    attribute_values = getattr(sampled, attribute_name)[idx_sample, :, :, idx_axis]
    range_ = getattr(sampled, attribute_name + '_range')
    # TODO:  this is a hack to account for transformed values that are nans, should be fixed when we have proper nan
    #  handling elsewhere in the code
    if attribute_name in ('trans_features', 'trans_responses', 'trans_predictions'):
        attribute_values = attribute_values.copy()
        attribute_values[attribute_values == sampled.data_sequence.nan_replacement_value] = np.nan
    x_label = '\n'.join(word.capitalize() for word in attribute_name.split('_') if word != 'raw').rstrip('s')
    min_, max_ = range_[idx_axis, :]
    ax.imshow(attribute_values, vmin=min_, vmax=max_)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('{} {}\n{}\n{}'.format(x_label, idx_axis, _format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))


def plot_softmax(sampled: samples.Samples, idx_sample: int, ax: plt.Axes, add_xlabel: bool, add_ylabel: bool) -> None:
    # Note:  this assumes that the softmax applied to all prediction axes and that there was no transformation applied
    #  to the categorical data.
    # TODO:  Phil:  are we going to have issues if a transformation was applied to categorical data? I think so?
    min_ = 0
    max_ = sampled.num_responses - 1
    assert not colormaps.check_is_categorical_colormap_repeated(sampled.num_responses), \
        'Number of categorical responses is greater than length of colormap, figure out how to handle gracefully'
    ax.imshow(np.argmax(sampled.raw_predictions[idx_sample, :], axis=-1), vmin=min_, vmax=max_,
              cmap=colormaps.COLORMAP_CATEGORICAL)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        # TODO:  Phil:  better label?
        ax.set_xlabel('Softmax\nCategories\n{}\n{}'.format(_format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))


def plot_weights(sampled: samples.Samples, idx_sample: int, ax: plt.Axes, add_xlabel: bool, add_ylabel: bool) -> None:
    min_, max_ = sampled.weights_range[0, :]
    weights = sampled.weights[idx_sample, :].copy()
    weights[weights == 0] = np.nan
    ax.imshow(weights, vmin=min_, vmax=max_, cmap=colormaps.COLORMAP_WEIGHTS)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Weights\n{}\n{}'.format(_format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))


def plot_error_categorical(
        sampled: samples.Samples,
        idx_sample: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    # Note:  this assumes that the softmax applied to all prediction axes and that there was no transformation applied
    #  to the categorical data.
    # TODO:  Phil:  are we going to have issues if a transformation was applied to categorical data? I think so?
    min_ = 0
    max_ = 1
    actual = np.argmax(sampled.raw_responses[idx_sample, :], axis=-1)
    predicted = np.argmax(sampled.raw_predictions[idx_sample, :], axis=-1)
    ax.imshow(predicted == actual, vmin=min_, vmax=max_, cmap=colormaps.COLORMAP_ERROR)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        # TODO:  Phil:  better label?
        ax.set_xlabel('Categorical\nErrors\n{}\n{}'.format(_format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))
    add_internal_window_to_subplot(sampled, ax)


def plot_raw_error_regression(
        sampled: samples.Samples,
        idx_sample: int,
        idx_response: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    error = sampled.raw_predictions[idx_sample, :, :, idx_response] - \
        sampled.raw_responses[idx_sample, :, :, idx_response]
    min_ = float(np.nanmin(error))
    max_ = float(np.nanmax(error))
    ax.imshow(error, vmin=min_, vmax=max_, cmap=colormaps.COLORMAP_ERROR)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Raw\nRegression\nErrors\n{}\n{}'.format(_format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))
    add_internal_window_to_subplot(sampled, ax)


def plot_transformed_error_regression(
        sampled: samples.Samples,
        idx_sample: int,
        idx_response: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    error = sampled.trans_predictions[idx_sample, :, :, idx_response] - \
        sampled.trans_responses[idx_sample, :, :, idx_response]
    max_ = float(np.max([np.abs(np.nanmin(error)), np.nanmax(error)]))
    min_ = - max_
    ax.imshow(error, vmin=min_, vmax=max_, cmap=colormaps.COLORMAP_ERROR)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Trans\nRegression\nErrors\n{}\n{}'.format(_format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))
    add_internal_window_to_subplot(sampled, ax)
    add_internal_window_to_subplot(sampled, ax)


def add_internal_window_to_subplot(sampled: samples.Samples, ax: plt.Axes) -> None:
    inshape = sampled.config.architecture.inshape
    loss_window_radius = sampled.config.data_build.loss_window_radius
    if (loss_window_radius * 2 == inshape[0]):
        return
    buffer = int((inshape[0] - loss_window_radius * 2) / 2)
    rect = patches.Rectangle(
        (buffer, buffer), loss_window_radius * 2, loss_window_radius * 2, linewidth=1, edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)


def _format_number(number: Union[int, float]) -> str:
    # isinstance needed for multiple numpy integer and float types
    if isinstance(number, int):
        return str(number)
    elif isinstance(number, float):
        return str(round(number, _FLOAT_DECIMALS))
