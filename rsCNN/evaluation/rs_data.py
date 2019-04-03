import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from rsCNN.data_management.sequences import BaseSequence


plt.switch_backend('Agg')  # Needed for remote server plotting


def plot_raw_and_scaled_input_examples(data_sequence: BaseSequence):
    features, responses = data_sequence.__getitem__(0)
    features = features[0]
    responses = responses[0]
    responses, weights = responses[..., :-1], responses[..., -1]

    features[features == data_sequence.feature_scaler.nodata_value] = np.nan
    responses[responses == data_sequence.response_scaler.nodata_value] = np.nan

    invtrans_features = data_sequence.feature_scaler.inverse_transform(features)
    trans_responses = data_sequence.response_scaler.transform(responses)

    fig_list = []
    # NOTE - this is not meant to be a universal config setup, which would be annoyingly hard.
    # This can always be exanded, but gives a reasonable amount of flexibility to start,
    # while showing the full range of things we should actually need to see.
    # Starting with the single response assumption.
    max_features_per_page = 3
    max_responses_per_page = 2

    max_samples_per_page = min(10, features.shape[0])
    max_pages = 8

    _feature_ind = 0
    _response_ind = 0
    _sample_ind = 0

    feat_mins, feat_maxs = _get_mins_maxs(features)
    resp_mins, resp_maxs = _get_mins_maxs(responses)
    invtrans_feat_mins, invtrans_feat_maxs = _get_mins_maxs(invtrans_features)
    trans_resp_mins, trans_resp_maxs = _get_mins_maxs(trans_responses)

    while _sample_ind < features.shape[0]:
        l_num_samp = min(max_samples_per_page, features.shape[0]-_sample_ind)

        while _feature_ind < features.shape[-1]:
            l_num_feat = min(max_features_per_page, features.shape[-1]-_feature_ind)
            l_num_resp = min(max_responses_per_page, responses.shape[-1]-_response_ind)

            fig = plt.figure(figsize=(30*((max_features_per_page + max_responses_per_page)*2 + 1) / ((max_features_per_page + max_responses_per_page)*2 + 1 + max_samples_per_page),
                                      30*max_samples_per_page / ((max_features_per_page + max_responses_per_page)*2 + 1 + max_samples_per_page)))
            gs1 = gridspec.GridSpec(l_num_samp, l_num_feat*2+l_num_resp*2+1)

            for _s in range(_sample_ind, _sample_ind + l_num_samp):
                for _f in range(_feature_ind, _feature_ind + l_num_feat):
                    ax = plt.subplot(gs1[_s-_sample_ind, _f-_feature_ind])
                    ax.imshow(np.squeeze(invtrans_features[_s, :, :, _f]),
                              vmin=invtrans_feat_mins[_f], vmax=invtrans_feat_maxs[_f])
                    plt.xticks([])
                    plt.yticks([])

                    if (_f == _feature_ind):
                        plt.ylabel('Sample ' + str(_s))
                    if (_s == _sample_ind):
                        plt.title('Feature ' + str(_f) + '\n' +
                                  str(round(invtrans_feat_mins[_f], 2)) + '\n' + str(round(invtrans_feat_maxs[_f], 2)))

                    ax = plt.subplot(gs1[_s-_sample_ind, l_num_feat + _f-_feature_ind])
                    ax.imshow(np.squeeze(features[_s, :, :, _f]),
                              vmin=feat_mins[_f], vmax=feat_maxs[_f])
                    plt.xticks([])
                    plt.yticks([])

                    if (_s == _sample_ind):
                        plt.title('Transformed\nFeature ' + str(_f) + '\n' +
                                  str(round(feat_mins[_f], 2)) + '\n' + str(round(feat_maxs[_f], 2)))

                for _r in range(_response_ind, _response_ind + l_num_resp):
                    ax = plt.subplot(gs1[_s-_sample_ind, 2*l_num_feat + _r-_response_ind])
                    ax.imshow(np.squeeze(responses[_s, :, :, _r]), vmin=resp_mins[_r], vmax=resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (_s == _sample_ind):
                        plt.title('Response ' + str(_r) + '\n' +
                                  str(round(resp_mins[_r], 2)) + '\n' + str(round(resp_maxs[_r], 2)))

                    ax = plt.subplot(gs1[_s-_sample_ind, 2*l_num_feat + l_num_resp + _r-_response_ind])
                    ax.imshow(np.squeeze(trans_responses[_s, :, :, _r]),
                              vmin=trans_resp_mins[_r], vmax=trans_resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (_s == _sample_ind):
                        plt.title('Transformed\nResponse ' + str(_r) + '\n' +
                                  str(round(trans_resp_mins[_r], 2)) + '\n' + str(round(trans_resp_maxs[_r], 2)))
                ax = plt.subplot(gs1[_s-_sample_ind, -1])
                ax.imshow(np.squeeze(weights[_s, :, :]), vmin=0, vmax=1, cmap='Greys_r')
                plt.xticks([])
                plt.yticks([])

                if (_s == _sample_ind):
                    plt.title('Weights')

            plt.suptitle('Input Example Plots Page ' + str((len(fig_list))))
            fig_list.append(fig)
            _feature_ind += max_features_per_page

            if (len(fig_list) > max_pages):
                break
        _sample_ind += max_samples_per_page
        if (len(fig_list) > max_pages):
            break

    return fig_list


def _get_mins_maxs(data):
    data_mins = np.nanpercentile(data.reshape((-1, data.shape[-1])), 0, axis=0)
    data_maxs = np.nanpercentile(data.reshape((-1, data.shape[-1])), 100, axis=0)
    return data_mins, data_maxs
