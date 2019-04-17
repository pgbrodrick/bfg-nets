from typing import Union

import matplotlib.pyplot as plt

from rsCNN.evaluation import samples


_COLORMAP_WEIGHTS = 'Greys_r'
_FLOAT_DECIMALS = 2


def plot_raw_features(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    return _plot_sample_attribute(sampled, idx_sample, idx_feature, 'raw_features', ax, add_xlabel, add_ylabel)


def plot_transformed_features(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    return _plot_sample_attribute(sampled, idx_sample, idx_feature, 'trans_features', ax, add_xlabel, add_ylabel)


def plot_raw_responses(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    return _plot_sample_attribute(sampled, idx_sample, idx_feature, 'raw_responses', ax, add_xlabel, add_ylabel)


def plot_transformed_responses(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    return _plot_sample_attribute(sampled, idx_sample, idx_feature, 'trans_responses', ax, add_xlabel, add_ylabel)


def _plot_sample_attribute(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        attribute_name: str,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    attribute = getattr(sampled, attribute_name)
    range_ = getattr(sampled, attribute_name + '_range')
    y_label = '\n'.join(word.capitalize() for word in attribute_name.split('_')).rstrip('s')
    min_, max_ = range_[idx_feature, :]
    ax.imshow(attribute[idx_sample, :, :, idx_feature], vmin=min_, vmax=max_)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Sample\n{}'.format(idx_sample))
    if add_ylabel:
        ax.set_ylabel('{}\n{}\n{}'.format(y_label, _format_number(min_), _format_number(max_)))


def plot_weights(sampled: samples.Samples, ax: plt.Axes, add_xlabel: bool) -> None:
    min_, max_ = sampled.weights_range[0, :]
    ax.imshow(sampled.weights, vmin=min_, vmax=max_, cmap=_COLORMAP_WEIGHTS)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Weights\n{}\n{}'.format(_format_number(min_), _format_number(max_)))


def _format_number(number: Union[int, float]) -> str:
    if type(number) is int:
        return str(number)
    elif type(number) is float:
        return str(round(number, _FLOAT_DECIMALS))
