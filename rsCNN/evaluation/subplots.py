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
    is_transformed = False
    return _plot_features(sampled, idx_sample, idx_feature, ax, is_transformed, add_xlabel, add_ylabel)


def plot_transformed_features(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        ax: plt.Axes,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    is_transformed = True
    return _plot_features(sampled, idx_sample, idx_feature, ax, is_transformed, add_xlabel, add_ylabel)


def _plot_features(
        sampled: samples.Samples,
        idx_sample: int,
        idx_feature: int,
        ax: plt.Axes,
        is_transformed: bool,
        add_xlabel: bool,
        add_ylabel: bool
) -> None:
    if is_transformed:
        features = sampled.trans_features
        range_ = sampled.trans_features_range
        label_prepend = 'Transformed\n'
    else:
        features = sampled.raw_features
        range_ = sampled.raw_features_range
        label_prepend = ''
    min_, max_ = range_[idx_feature, :]
    ax.imshow(features[idx_sample, :, :, idx_feature], vmin=min_, vmax=max_)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_xlabel:
        ax.set_xlabel('Sample\n{}'.format(idx_sample))
    if add_ylabel:
        ax.set_ylabel('{}Feature\n{}\n{}'.format(label_prepend, _format_number(min_), _format_number(max_)))


def plot_weights(sampled: samples.Samples, ax: plt.Axes, add_xlabel: bool = False) -> None:
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
