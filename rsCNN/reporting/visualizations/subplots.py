import logging
from typing import Union

from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

from rsCNN.reporting import samples
from rsCNN.reporting.visualizations import colormaps


plt.switch_backend('Agg')  # Needed for remote server plotting

_logger = logging.getLogger(__name__)

_LABEL_CATEGORICAL_RESPONSES = 'categorical_responses'
_FLOAT_DECIMALS = 2

"""    
Note:  we need to know how many plots to generate for some reports, so that we can specify the number of columns in a 
figure. We want to keep the actual plotting code synced to the code that calculates the number of columns to avoid 
bugs. However, we also do not want to be writing a ton of boilerplate code to check whether plots should be created. 
The repeated check for `if ax is not None` is a quick and easy way to avoid plotting when simply trying to get the 
number of columns in a single place, even though it feels like it's in the wrong place. This also makes the code a
bit more readable, but I realize this is subjective. Better solutions are welcomed.
"""


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


def plot_categorical_responses(
        sampled: samples.Samples,
        idx_sample: int,
        idx_response: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    _plot_sample_attribute(sampled, idx_sample, idx_response, _LABEL_CATEGORICAL_RESPONSES, ax, add_xlabel, add_ylabel)


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
    # Return early if axis is not provided or attribute is not available
    if ax is None:
        return
    if attribute_name != _LABEL_CATEGORICAL_RESPONSES:
        is_attribute_available = getattr(sampled, attribute_name) is not None
    else:
        is_attribute_available = sampled.raw_responses is not None
    if not is_attribute_available:
        _logger.debug('Not plotting {}; attribute not available'.format(
            attribute_name if attribute_name != _LABEL_CATEGORICAL_RESPONSES else 'raw_responses'))
        return

    # Categorical responses need special handling, in that we use a particular color scheme
    if attribute_name != _LABEL_CATEGORICAL_RESPONSES:
        attribute_values = getattr(sampled, attribute_name)[idx_sample, :, :, idx_axis]
        min_, max_ = getattr(sampled, attribute_name + '_range')[idx_axis, :]
        colormap = colormaps.COLORMAP_DEFAULT
    else:
        attribute_values = np.argmax(sampled.raw_responses[idx_sample, ...], axis=-1)
        min_ = 0
        max_ = sampled.num_responses - 1
        colormap = colormaps.COLORMAP_CATEGORICAL
        assert not colormaps.check_is_categorical_colormap_repeated(sampled.num_responses), \
            'Number of categorical responses is greater than length of colormap, figure out how to handle gracefully'

    # Handle nan conversions for transformed data
    if attribute_name in ('trans_features', 'trans_responses', 'trans_predictions'):
        attribute_values = attribute_values.copy()
        attribute_values[attribute_values == sampled.data_sequence.nan_replacement_value] = np.nan

    # Plot
    ax.imshow(attribute_values, vmin=min_, vmax=max_, cmap=colormap)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        x_label = '\n'.join(word.capitalize() for word in attribute_name.split('_') if word != 'raw').rstrip('s')
        ax.set_xlabel(
            '{} {}\n{}\n{}'.format(x_label, idx_axis, _format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))


def plot_classification_predictions_max_likelihood(
        sampled: samples.Samples,
        idx_sample: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    if ax is None:
        return
    if sampled.raw_predictions is None:
        _logger.debug('Not plotting raw predictions max likelihood; raw_predictions attribute not available')
        return

    # Note:  this assumes that the softmax applied to all prediction axes and that there was no transformation applied
    #  to the categorical data.
    min_ = 0
    max_ = sampled.num_responses - 1
    assert not colormaps.check_is_categorical_colormap_repeated(sampled.num_responses), \
        'Number of categorical responses is greater than length of colormap, figure out how to handle gracefully'
    ax.imshow(np.argmax(sampled.raw_predictions[idx_sample, ...], axis=-1), vmin=min_, vmax=max_,
              cmap=colormaps.COLORMAP_CATEGORICAL)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Categorical\nPredictions MLE\n{}\n{}'.format(_format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))


def plot_weights(sampled: samples.Samples, idx_sample: int, ax: plt.Axes, add_xlabel: bool, add_ylabel: bool) -> None:
    if ax is None:
        return
    if sampled.weights is None:
        _logger.debug('Not plotting weights; weights attribute not available')
        return

    min_, max_ = sampled.weights_range[0, :]
    weights = sampled.weights[idx_sample, :].copy()
    weights[weights == 0] = np.nan

    weight_cmap = matplotlib.cm.get_cmap(colormaps.COLORMAP_WEIGHTS)
    weight_cmap.set_bad('white')

    ax.imshow(weights, vmin=min_, vmax=max_, cmap=weight_cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Weights\n{}\n{}'.format(_format_number(min_), _format_number(max_)))
        ax.xaxis.set_label_position('top')
    if add_ylabel:
        ax.set_ylabel('Sample\n{}'.format(idx_sample))


def plot_binary_error_classification(
        sampled: samples.Samples,
        idx_sample: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    if ax is None:
        return
    if sampled.raw_responses is None or sampled.raw_predictions is None:
        _logger.debug('Not plotting classification errors; ' +
                      'raw_responses and/or raw_predictions attributes are not available')
        return

    # Note:  this assumes that the softmax applied to all prediction axes and that there was no transformation applied
    #  to the categorical data.
    # Note:  the actual range of this data will be from 0 to 1, i.e., is the class incorrect or correct, but the plots
    #  will be too dark if we set the vmin and vmax to 0 and 1, respectively
    min_ = -0.5
    max_ = 1.5
    actual = np.argmax(sampled.raw_responses[idx_sample, :], axis=-1)
    predicted = np.argmax(sampled.raw_predictions[idx_sample, :], axis=-1)
    ax.imshow(predicted == actual, vmin=min_, vmax=max_, cmap=colormaps.COLORMAP_ERROR)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Correct\nCategory\nFalse\nTrue')
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
    if ax is None:
        return
    if sampled.raw_responses is None or sampled.raw_predictions is None:
        _logger.debug('Not plotting raw regression errors; ' +
                      'raw_responses and/or raw_predictions attributes are not available')
        return

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
    if ax is None:
        return
    if sampled.raw_responses is None or sampled.raw_predictions is None:
        _logger.debug('Not plotting transformed regression errors; ' +
                      'raw_responses and/or raw_predictions attributes are not available')
        return

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
    if ax is None:
        return

    loss_window_radius = sampled.config.data_build.loss_window_radius
    window_radius = sampled.config.data_build.window_radius
    if loss_window_radius == window_radius:
        return
    buffer = int(window_radius - loss_window_radius)
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
