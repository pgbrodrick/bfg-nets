from typing import List

import keras
import matplotlib.pyplot as plt
import numpy as np

from rsCNN.evaluation import samples


plt.switch_backend('Agg')  # Needed for remote server plotting


def print_model_summary(model: keras.Model) -> plt.Figure:
    stringlist = ['CNN Architecture Summary']
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary_string = "\n".join(stringlist)

    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
    plt.text(0, 0, model_summary_string, **{'fontsize': 8, 'fontfamily': 'monospace'})
    plt.axis('off')

    return fig


def plot_network_feature_progression(
        sampled: samples.Samples,
        compact: bool = False,
        max_pages: int = 10,
        max_filters: int = 10
) -> List[plt.Figure]:
    return [_plot_sample_feature_progression(sampled, idx_sample, compact, max_filters)
            for idx_sample in range(min(max_pages, sampled.num_samples))]


def _plot_sample_feature_progression(
        sampled: samples.Samples,
        idx_sample: int,
        compact: bool,
        max_filters: int
) -> plt.Figure:
    sample_features = sampled.trans_features or sampled.raw_features
    sample_features = np.expand_dims(sample_features[idx_sample, :], 0)
    sample_responses = sampled.trans_responses or sampled.raw_responses
    sample_responses = np.expand_dims(sample_responses[idx_sample, :], 0)

    # Run through the model and grab any Conv2D layer (other layers could also be grabbed as desired)
    pred_set = []
    layer_names = []
    pred_set.append(sample_features)
    layer_names.append('Feature(s)')
    for _l in range(0, len(sampled.model.layers)):
        if (isinstance(sampled.model.layers[_l], keras.layers.convolutional.Conv2D)):
            im_model = keras.models.Model(
                inputs=sampled.model.layers[0].output, outputs=sampled.model.layers[_l].output)
            pred_set.append(im_model.predict(sample_features))
            layer_names.append(sampled.model.layers[_l].name)
    pred_set.append(sample_responses)
    layer_names.append('Response(s)')

    # Calculate the per-filter standard deviation, enables plots to preferentially show more interesting layers
    pred_std = []
    for _l in range(len(pred_set)):
        pred_std.append([np.std(np.squeeze(pred_set[_l][..., x])) for x in range(0, pred_set[_l].shape[-1])])

    # Get spacing things worked out and the figure initialized
    step_size = 1 / float(len(pred_set)+1)
    if (compact):
        h_space_fraction = 0.3
    else:
        h_space_fraction = 0.05

    image_size = min(step_size * (1-h_space_fraction), 1 / max_filters * (1-h_space_fraction))
    h_space_size = step_size*h_space_fraction

    fig = plt.figure(figsize=(max(max_filters, len(pred_set)), max(max_filters, len(pred_set))))

    top = 0
    # Step through each layer in the network
    for _l in range(0, len(pred_set)):
        # Step through each filter, up to the max
        for _iii in range(0, min(pred_set[_l].shape[-1], max_filters)):

            if (compact):
                ip = [(_l+0.5)*step_size + _iii*h_space_size/5., _iii*image_size*0.2]
            else:
                ip = [(_l+0.5)*step_size, _iii*image_size*(1+h_space_fraction)]

            # Get the indices sorted by filter std, as a proxy for interest
            ordered_pred_std = np.argsort(pred_std[_l])[::-1]
            # prep the image
            tp = np.squeeze(pred_set[_l][:, ordered_pred_std[_iii]])

            # Plot!
            ax = fig.add_axes([ip[0], ip[1], image_size, image_size], zorder=max_filters+1-_iii)
            top = max(top, ip[1]+image_size)
            plt.imshow(tp, vmin=np.nanpercentile(tp, 0), vmax=np.nanpercentile(tp, 100))
            _adjust_axis(ax)
            if (_iii == 0):
                plt.xlabel(layer_names[_l])

    tit = 'Network Feature Progression Visualization {}'.format(idx_sample)
    if (compact):
        tit = 'Compact ' + tit
    ax = fig.add_axes([0.5, top + image_size/2., 0.01, 0.01], zorder=100)
    ax.axis('off')
    ax.text(0, 0, tit, ha='center', va='center')
    return fig


def _adjust_axis(lax):
    for sp in lax.spines:
        lax.spines[sp].set_color('white')
        lax.spines[sp].set_linewidth(2)
    lax.set_xticks([])
    lax.set_yticks([])
