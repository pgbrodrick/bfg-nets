import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.networks.experiment import Experiment


# TODO:  I don't know everything this function does, but we can probably make it more specific. We have
#  plot_raw_and_scaled_input_examples for input data... is this something like plot_raw_and_scaled_result_examples?

def plot_predictions(data_sequence: BaseSequence, experiment: Experiment):
    features, responses = data_sequence.__getitem__(0)
    responses, weights = responses[..., :-1], responses[..., -1]
    predictions = experiment.predict(features)

    features[features == data_sequence.feature_scaler.nodata_value] = np.nan
    responses[responses == data_sequence.response_scaler.nodata_value] = np.nan

    raw_responses = data_sequence.response_scaler.inverse_transform(responses)
    raw_predictions = data_sequence.response_scaler.inverse_transform(predictions)

    internal_window_radius = experiment.data_config.internal_window_radius

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
    #invtrans_pred_resp_mins, invtrans_pred_resp_maxs = _get_mins_maxs(raw_predictions)
    invtrans_pred_resp_mins, invtrans_pred_resp_maxs = _get_mins_maxs(raw_responses)

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
                ax.imshow(np.squeeze(weights[_s, :, :]), vmin=0, vmax=1, cmap='Greys_r')
                plt.xticks([])
                plt.yticks([])

                if (_s == _sample_ind):
                    plt.title('Weights')

            plt.suptitle('Prediction Plots Page ' + str((len(fig_list))))
            fig_list.append(fig)
            _response_ind += max_responses_per_page

            if (len(fig_list) > max_pages):
                break
        _sample_ind += max_samples_per_page
        if (len(fig_list) > max_pages):
            break

    return fig_list


# TODO:  update name, histograms is a bit unclear as to what this actually plots
# TODO:  I started refactoring this, but I don't quite know what fa is and whether we need fa or verification_fold. I'm
#  guessing that we can get both fa and verification_fold from the experiment object, as in the plotting function above,
#  or that we can simply use the verification sequence. Taking a step back, it seems like it might be useful to make a
#  class that looks like the following, and then the plotting functions are methods on this object that have access to
#  the features, responses, and predictions without recreating those objects and without passing a shitton of variables
#  around.

class ResultsReport(object):

    def __init__(self, data_sequence: BaseSequence, experiment: Experiment):
        self.data_sequence = data_sequence
        self.experiment = experiment
        self.features, tmp_responses = self.data_sequence.__getitem__(0)
        self.responses, self.weights = tmp_responses[..., :-1], tmp_responses[..., -1]
        self.predictions = experiment.predict(self.features)

        self.features[self.features == data_sequence.feature_scaler.nodata_value] = np.nan
        self.responses[self.responses == data_sequence.response_scaler.nodata_value] = np.nan

        self.raw_responses = data_sequence.response_scaler.inverse_transform(self.responses)
        self.raw_predictions = data_sequence.response_scaler.inverse_transform(self.predictions)

    # TODO:  move plotting functions to methods on this object


def plot_prediction_histograms(responses, weights, pred_responses, fa, verification_fold, response_transform, response_nodata_value):

    fig_list = []

    fold_assignments = np.zeros((responses.shape[0], responses.shape[1], responses.shape[2], 1))
    un_fa = np.unique(fa)
    for _n in range(len(un_fa)):
        fold_assignments[fa == un_fa[_n], ...] = un_fa[_n]

    responses[responses == response_nodata_value] = np.nan
    pred_responses[pred_responses == response_nodata_value] = np.nan

    trans_responses = response_transform.transform(responses)
    invtrans_pred_responses = response_transform.inverse_transform(pred_responses)

    responses[weights == 0, :] = np.nan
    pred_responses[weights == 0, :] = np.nan
    trans_responses[weights == 0, :] = np.nan
    invtrans_pred_responses[weights == 0, :] = np.nan

    responses = responses.reshape((-1, responses.shape[-1]))
    pred_responses = pred_responses.reshape((-1, pred_responses.shape[-1]))
    trans_responses = trans_responses.reshape((-1, trans_responses.shape[-1]))
    invtrans_pred_responses = invtrans_pred_responses.reshape((-1, invtrans_pred_responses.shape[-1]))
    fold_assignments = np.squeeze(fold_assignments.reshape((-1, fold_assignments.shape[-1])))

    max_resp_per_page = min(8, responses.shape[-1])
    _response_ind = 0

    # Training Raw Space
    while _response_ind < responses.shape[-1]:

        fig = plt.figure(figsize=(30 * max_resp_per_page / (4 + max_resp_per_page), 30 * 4 / (4 + max_resp_per_page)))
        gs1 = gridspec.GridSpec(4, max_resp_per_page)
        for _r in range(_response_ind, min(_response_ind+max_resp_per_page, responses.shape[-1])):
            ax = plt.subplot(gs1[0, _r])
            plt.hist(responses[fold_assignments != verification_fold, _r], color='black')
            plt.hist(invtrans_pred_responses[fold_assignments != verification_fold, _r], color='green')

            if (_r == _response_ind):
                plt.ylabel('Training\nRaw')
            plt.title('Response ' + str(_r))

            ax = plt.subplot(gs1[1, _r])
            plt.hist(responses[fold_assignments == verification_fold, _r], color='black')
            plt.hist(invtrans_pred_responses[fold_assignments == verification_fold, _r], color='green')

            if (_r == _response_ind):
                plt.ylabel('Testing\nRaw')

            ax = plt.subplot(gs1[2, _r])
            plt.hist(trans_responses[fold_assignments != verification_fold, _r], color='black')
            plt.hist(pred_responses[fold_assignments != verification_fold, _r], color='green')

            if (_r == _response_ind):
                plt.ylabel('Training\nTransform')

            ax = plt.subplot(gs1[3, _r])
            plt.hist(trans_responses[fold_assignments == verification_fold, _r], color='black')
            plt.hist(pred_responses[fold_assignments == verification_fold, _r], color='green')

            if (_r == _response_ind):
                plt.ylabel('Testing\nRaw')

        _response_ind += max_resp_per_page
        plt.suptitle('Response Histogram Page ' + str((len(fig_list))))
        fig_list.append(fig)
    return fig_list


def spatial_error(responses, pred_responses, weights, response_nodata_value):

    # TODO: Consider handling weights
    fig_list = []

    responses[responses == response_nodata_value] = np.nan
    pred_responses[responses == response_nodata_value] = np.nan

    diff = np.abs(responses-pred_responses)

    max_resp_per_page = min(8, responses.shape[-1])

    _response_ind = 0
    while (_response_ind < responses.shape[-1]):
        fig = plt.figure(figsize=(25, 8))
        gs1 = gridspec.GridSpec(1, max_resp_per_page)
        for _r in range(_response_ind, min(responses.shape[-1], _response_ind + max_resp_per_page)):
            ax = plt.subplot(gs1[0, _r - _response_ind])

            vset = np.nanmean(diff[..., _r]*weights, axis=0)
            ax.imshow(np.nanmean(diff[..., _r], axis=0), vmin=np.nanmin(
                vset[vset != 0]), vmax=np.nanmax(vset[vset != 0]))
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
