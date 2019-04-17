import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import samples, subplots


def plot_raw_and_scaled_result_examples(
        model: keras.Model,
        network_config: dict,
        data_sequence: BaseSequence
):
    sampled = samples.Samples(data_sequence, model, network_config)
    internal_window_radius = network_config['architecture']['internal_window_radius']

    fig_list = []
    # NOTE - this is not meant to be a universal config setup, which would be annoyingly hard.
    # This can always be exanded, but gives a reasonable amount of flexibility to start,
    # while showing the full range of things we should actually need to see.
    # Starting with the single response assumption.
    max_responses_per_page = 1

    max_samples_per_page = min(10, sampled.num_samples)
    max_pages = 8

    _response_ind = 0
    _sample_ind = 0

    while _sample_ind < sampled.num_samples:
        l_num_samp = min(max_samples_per_page, sampled.num_samples - _sample_ind)

        while _response_ind < sampled.num_responses:
            l_num_resp = min(max_responses_per_page, sampled.num_responses - _response_ind)

            fig = plt.figure(figsize=(30*(max_responses_per_page*4 + 1) / (max_responses_per_page*4 + 1 + max_samples_per_page),
                                      30*max_samples_per_page / (max_responses_per_page*4 + 1 + max_samples_per_page)))
            gs1 = gridspec.GridSpec(l_num_samp, l_num_resp*4+1)

            for _s in range(_sample_ind, _sample_ind + l_num_samp):
                for _r in range(_response_ind, _response_ind + l_num_resp):
                    # Raw response
                    ax = plt.subplot(gs1[_s-_sample_ind, _r-_response_ind])
                    subplots.plot_raw_responses(sampled, _s, _r, ax, _s == _sample_ind, _r == _response_ind)

                    # Transformed response
                    ax = plt.subplot(gs1[_s-_sample_ind, l_num_resp + _r-_response_ind])
                    subplots.plot_transformed_responses(sampled, _s, _r, ax, False, _r == _response_ind)

                    # Prediction
                    ax = plt.subplot(gs1[_s-_sample_ind, 2*l_num_resp + _r-_response_ind])
                    subplots.plot_raw_predictions(sampled, _s, _r, ax, False, _s == _sample_ind)

                    # Transformed Prediction
                    ax = plt.subplot(gs1[_s-_sample_ind, 3*l_num_resp + _r-_response_ind])
                    subplots.plot_transformed_predictions(sampled, _s, _r, ax, False, _s == _sample_ind)

                ax = plt.subplot(gs1[_s-_sample_ind, -1])
                subplots.plot_weights(sampled, ax, _s == _sample_ind)

            plt.suptitle('Prediction Plots Page ' + str((len(fig_list))))
            fig_list.append(fig)
            _response_ind += max_responses_per_page

            if (len(fig_list) > max_pages):
                break
        _sample_ind += max_samples_per_page
        if (len(fig_list) > max_pages):
            break

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
