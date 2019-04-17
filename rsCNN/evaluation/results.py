import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import subplots


def plot_raw_and_scaled_result_examples(
        model: keras.Model,
        network_config: dict,
        data_sequence: BaseSequence
):
    features, responses = data_sequence.__getitem__(0)
    predictions = model.predict(features)
    features = features[0]
    responses = responses[0]
    responses, weights = responses[..., :-1], responses[..., -1]
    weights = weights.reshape((weights.shape[0], weights.shape[1], weights.shape[2], 1))

    features[features == data_sequence.feature_scaler.nodata_value] = np.nan
    responses[responses == data_sequence.response_scaler.nodata_value] = np.nan

    raw_responses = data_sequence.response_scaler.inverse_transform(responses)
    raw_predictions = data_sequence.response_scaler.inverse_transform(predictions)

    internal_window_radius = network_config['architecture']['internal_window_radius']

    fig_list = []
    # NOTE - this is not meant to be a universal config setup, which would be annoyingly hard.
    # This can always be exanded, but gives a reasonable amount of flexibility to start,
    # while showing the full range of things we should actually need to see.
    # Starting with the single response assumption.
    max_responses_per_page = 1

    max_samples_per_page = min(10, raw_responses.shape[0])
    max_pages = 8

    _response_ind = 0
    _sample_ind = 0

    resp_mins, resp_maxs = _get_mins_maxs(raw_responses)
    trans_resp_mins, trans_resp_maxs = _get_mins_maxs(responses)
    pred_resp_mins, pred_resp_maxs = _get_mins_maxs(predictions)
    invtrans_pred_resp_mins, invtrans_pred_resp_maxs = _get_mins_maxs(raw_responses)
    weight_mins, weight_maxs = _get_mins_maxs(weights)

    while _sample_ind < raw_responses.shape[0]:
        l_num_samp = min(max_samples_per_page, raw_responses.shape[0]-_sample_ind)

        while _response_ind < raw_responses.shape[-1]:
            l_num_resp = min(max_responses_per_page, raw_responses.shape[-1]-_response_ind)

            fig = plt.figure(figsize=(30*(max_responses_per_page*4 + 1) / (max_responses_per_page*4 + 1 + max_samples_per_page),
                                      30*max_samples_per_page / (max_responses_per_page*4 + 1 + max_samples_per_page)))
            gs1 = gridspec.GridSpec(l_num_samp, l_num_resp*4+1)

            for _s in range(_sample_ind, _sample_ind + l_num_samp):
                for _r in range(_response_ind, _response_ind + l_num_resp):
                    # Raw response
                    ax = plt.subplot(gs1[_s-_sample_ind, _r-_response_ind])
                    ax.imshow(np.squeeze(raw_responses[_s, :, :, _r]), vmin=resp_mins[_r], vmax=resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (_r == _response_ind):
                        plt.ylabel('Sample ' + str(_s))
                    if (_s == _sample_ind):
                        plt.title('Response ' + str(_r) + '\n' +
                                  str(round(resp_mins[_r], 2)) + '\n' + str(round(resp_maxs[_r], 2)))

                    # Transformed response
                    ax = plt.subplot(gs1[_s-_sample_ind, l_num_resp + _r-_response_ind])
                    ax.imshow(np.squeeze(responses[_s, :, :, _r]),
                              vmin=trans_resp_mins[_r], vmax=trans_resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (_s == _sample_ind):
                        plt.title('Transformed\nResponse ' + str(_r) + '\n' +
                                  str(round(trans_resp_mins[_r], 2)) + '\n' + str(round(trans_resp_maxs[_r], 2)))

                    # Prediction
                    ax = plt.subplot(gs1[_s-_sample_ind, 2*l_num_resp + _r-_response_ind])
                    ax.imshow(np.squeeze(predictions[_s, :, :, _r]),
                              vmin=pred_resp_mins[_r], vmax=pred_resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (internal_window_radius*2 != raw_responses.shape[1]):
                        buf = (raw_responses.shape[1] - internal_window_radius*2)/2
                        rect = patches.Rectangle((buf, buf), internal_window_radius*2,
                                                 internal_window_radius*2, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                    if (_s == _sample_ind):
                        plt.title('Predicted\nResponse ' + str(_r) + '\n' +
                                  str(round(pred_resp_mins[_r], 2)) + '\n' + str(round(pred_resp_maxs[_r], 2)))

                    # Transformed Prediction
                    ax = plt.subplot(gs1[_s-_sample_ind, 3*l_num_resp + _r-_response_ind])
                    ax.imshow(np.squeeze(raw_predictions[_s, :, :, _r]),
                              vmin=invtrans_pred_resp_mins[_r], vmax=invtrans_pred_resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (_s == _sample_ind):
                        plt.title('Inverse transform\nPredicted\nResponse ' + str(_r) + '\n' +
                                  str(round(invtrans_pred_resp_mins[_r], 2)) + '\n' + str(round(invtrans_pred_resp_maxs[_r], 2)))

                    if (internal_window_radius*2 != raw_responses.shape[1]):
                        buf = (raw_responses.shape[1] - internal_window_radius*2)/2
                        rect = patches.Rectangle((buf, buf), internal_window_radius*2,
                                                 internal_window_radius*2, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                ax = plt.subplot(gs1[_s-_sample_ind, -1])
                subplots.plot_weights(weights[_s, :, :], ax, weight_mins[0], weight_maxs[0])

                if (_s == _sample_ind):
                    plt.title('Weights = \n' +
                              str(round(weight_mins[0], 2)) + '\n' + str(round(weight_maxs[0], 2)))

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


def _get_mins_maxs(data):
    data_mins = np.nanpercentile(data.reshape((-1, data.shape[-1])), 0, axis=0)
    data_maxs = np.nanpercentile(data.reshape((-1, data.shape[-1])), 100, axis=0)
    return data_mins, data_maxs
