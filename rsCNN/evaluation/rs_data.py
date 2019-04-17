import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import samples, shared


plt.switch_backend('Agg')  # Needed for remote server plotting


def plot_raw_and_scaled_input_examples(sampled: samples.Samples):
    fig_list = []
    # NOTE - this is not meant to be a universal config setup, which would be annoyingly hard.
    # This can always be exanded, but gives a reasonable amount of flexibility to start,
    # while showing the full range of things we should actually need to see.
    # Starting with the single response assumption.
    max_features_per_page = 5
    max_responses_per_page = 5

    max_samples_per_page = min(10, sampled.num_samples)
    max_pages = 8

    _feature_ind = 0
    _response_ind = 0
    _sample_ind = 0

    while _sample_ind < sampled.num_samples:
        l_num_samp = min(max_samples_per_page, sampled.num_samples-_sample_ind)

        while _feature_ind < sampled.num_features:
            l_num_feat = min(max_features_per_page, sampled.num_features-_feature_ind)
            l_num_resp = min(max_responses_per_page, sampled.num_responses-_response_ind)

            fig = plt.figure(figsize=(30*((max_features_per_page + max_responses_per_page)*2 + 1) / ((max_features_per_page + max_responses_per_page)*2 + 1 + max_samples_per_page),
                                      30*max_samples_per_page / ((max_features_per_page + max_responses_per_page)*2 + 1 + max_samples_per_page)))
            gs1 = gridspec.GridSpec(l_num_samp, l_num_feat*2+l_num_resp*2+1)

            for _s in range(_sample_ind, _sample_ind + l_num_samp):
                for _f in range(_feature_ind, _feature_ind + l_num_feat):
                    ax = plt.subplot(gs1[_s-_sample_ind, _f-_feature_ind])
                    shared.plot_raw_features(sampled, _s, _f, ax, _f == _feature_ind, _s == _sample_ind)
                    ax = plt.subplot(gs1[_s - _sample_ind, l_num_feat + _f - _feature_ind])
                    shared.plot_transformed_features(sampled, _s, _f, ax, False, _s == _sample_ind)

                for _r in range(_response_ind, _response_ind + l_num_resp):
                    ax = plt.subplot(gs1[_s-_sample_ind, 2*l_num_feat + _r-_response_ind])
                    shared.plot_raw_responses(sampled, _s, _f, ax, False, _s == _sample_ind)
                    ax = plt.subplot(gs1[_s-_sample_ind, 2*l_num_feat + l_num_resp + _r-_response_ind])
                    shared.plot_transformed_responses(sampled, _s, _f, ax, False, _s == _sample_ind)
                ax = plt.subplot(gs1[_s-_sample_ind, -1])
                shared.plot_weights(sampled, ax, _s == _sample_ind)

            plt.suptitle('Input Example Plots Page ' + str((len(fig_list))))
            fig_list.append(fig)
            _feature_ind += max_features_per_page

            if (len(fig_list) > max_pages):
                break
        _sample_ind += max_samples_per_page
        if (len(fig_list) > max_pages):
            break

    return fig_list
