import numpy as np
import os
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.use('Agg')


def plot_history(history):

    fig = plt.figure(figsize=(13, 10))
    gs1 = gridspec.GridSpec(2, 2)

    # Epoch times and delays
    ax = plt.subplot(gs1[0, 0])
    epoch_time = [(finish - start).seconds for start, finish in zip(history['epoch_start'], history['epoch_finish'])]
    epoch_delay = [(start - finish).seconds for start, finish
                   in zip(history['epoch_start'][1:], history['epoch_finish'][:-1])]
    ax.plot(epoch_time, c='black', label='Epoch time')
    ax.plot(epoch_delay, '--', c='blue', label='Epoch delay')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Seconds')
    ax.legend()

    # Epoch times different view
    ax = plt.subplot(gs1[0, 1])
    dts = [epoch.strftime('%d %H:%M') for epoch in history['epoch_finish']]
    ax.hist(dts)
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylabel('Epochs completed')

    # Loss
    ax = plt.subplot(gs1[1, 0])
    ax.plot(history['loss'][-160:], c='black', label='Training loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'][-160:], '--', c='blue', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Learning rate
    ax = plt.subplot(gs1[1, 1])
    ax.plot(history['lr'][-160:], c='black')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')

    # Add figure title
    plt.suptitle('Training History')
    return fig


def plot_model_summary_as_fig(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary_string = "\n".join(stringlist)

    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
    plt.text(0, 0, model_summary_string, **{'fontsize': 8, 'fontfamily': 'monospace'})
    plt.axis('off')
    plt.suptitle('CNN Summary')

    return fig


def get_mins_maxs(data):
    data_mins = np.nanpercentile(data.reshape((-1, data.shape[-1])), 0, axis=0)
    data_maxs = np.nanpercentile(data.reshape((-1, data.shape[-1])), 100, axis=0)
    return data_mins, data_maxs


def plot_input_examples_and_transforms(features, responses, weights, feature_transform, response_transform, feature_nodata_value, response_nodata_value):

    features[features == feature_nodata_value] = np.nan
    responses[responses == response_nodata_value] = np.nan

    trans_features = feature_transform.transform(features)
    trans_responses = response_transform.transform(responses)

    fig_list = []
    # NOTE - this is not meant to be a universal config setup, which would be annoyingly hard.
    # This can always be exanded, but gives a reasonable amount of flexibility to start,
    # while showing the full range of things we should actually need to see.
    # Starting with the single response assumption.
    max_features_per_page = 3
    max_responses_per_page = 1

    max_samples_per_page = min(10, features.shape[0])
    max_pages = 8

    _feature_ind = 0
    _response_ind = 0
    _sample_ind = 0

    feat_mins, feat_maxs = get_mins_maxs(features)
    resp_mins, resp_maxs = get_mins_maxs(responses)
    trans_feat_mins, trans_feat_maxs = get_mins_maxs(trans_features)
    trans_resp_mins, trans_resp_maxs = get_mins_maxs(trans_responses)

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
                    ax.imshow(np.squeeze(features[_s, :, :, _f]), vmin=feat_mins[_f], vmax=feat_maxs[_f])
                    plt.xticks([])
                    plt.yticks([])

                    if (_f == _feature_ind):
                        plt.ylabel('Sample ' + str(_s))
                    if (_s == _sample_ind):
                        plt.title('Feature ' + str(_f) + '\n' +
                                  str(round(feat_mins[_f], 2)) + '\n' + str(round(feat_maxs[_f], 2)))

                    ax = plt.subplot(gs1[_s-_sample_ind, l_num_feat + _f-_feature_ind])
                    ax.imshow(np.squeeze(trans_features[_s, :, :, _f]),
                              vmin=trans_feat_mins[_f], vmax=trans_feat_maxs[_f])
                    plt.xticks([])
                    plt.yticks([])

                    if (_s == _sample_ind):
                        plt.title('Transformed\nFeature ' + str(_f) + '\n' +
                                  str(round(trans_feat_mins[_f], 2)) + '\n' + str(round(trans_feat_maxs[_f], 2)))

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


def plot_predictions(responses, weights, pred_responses, response_transform, response_nodata_value, internal_window_radius):

    responses[responses == response_nodata_value] = np.nan
    pred_responses[pred_responses == response_nodata_value] = np.nan

    trans_responses = response_transform.transform(responses)
    invtrans_pred_responses = response_transform.inverse_transform(pred_responses)

    fig_list = []
    # NOTE - this is not meant to be a universal config setup, which would be annoyingly hard.
    # This can always be exanded, but gives a reasonable amount of flexibility to start,
    # while showing the full range of things we should actually need to see.
    # Starting with the single response assumption.
    max_responses_per_page = 1

    max_samples_per_page = min(10, responses.shape[0])
    max_pages = 8

    _response_ind = 0
    _sample_ind = 0

    resp_mins, resp_maxs = get_mins_maxs(responses)
    trans_resp_mins, trans_resp_maxs = get_mins_maxs(trans_responses)
    pred_resp_mins, pred_resp_maxs = get_mins_maxs(pred_responses)
    #invtrans_pred_resp_mins, invtrans_pred_resp_maxs = get_mins_maxs(invtrans_pred_responses)
    invtrans_pred_resp_mins, invtrans_pred_resp_maxs = get_mins_maxs(responses)

    while _sample_ind < responses.shape[0]:
        l_num_samp = min(max_samples_per_page, responses.shape[0]-_sample_ind)

        while _response_ind < responses.shape[-1]:
            l_num_resp = min(max_responses_per_page, responses.shape[-1]-_response_ind)

            fig = plt.figure(figsize=(30*(max_responses_per_page*4 + 1) / (max_responses_per_page*4 + 1 + max_samples_per_page),
                                      30*max_samples_per_page / (max_responses_per_page*4 + 1 + max_samples_per_page)))
            gs1 = gridspec.GridSpec(l_num_samp, l_num_resp*4+1)

            for _s in range(_sample_ind, _sample_ind + l_num_samp):
                for _r in range(_response_ind, _response_ind + l_num_resp):
                    # Raw response
                    ax = plt.subplot(gs1[_s-_sample_ind, _r-_response_ind])
                    ax.imshow(np.squeeze(responses[_s, :, :, _r]), vmin=resp_mins[_r], vmax=resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (_r == _response_ind):
                        plt.ylabel('Sample ' + str(_s))
                    if (_s == _sample_ind):
                        plt.title('Response ' + str(_r) + '\n' +
                                  str(round(resp_mins[_r], 2)) + '\n' + str(round(resp_maxs[_r], 2)))

                    # Transformed response
                    ax = plt.subplot(gs1[_s-_sample_ind, l_num_resp + _r-_response_ind])
                    ax.imshow(np.squeeze(trans_responses[_s, :, :, _r]),
                              vmin=trans_resp_mins[_r], vmax=trans_resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (_s == _sample_ind):
                        plt.title('Transformed\nResponse ' + str(_r) + '\n' +
                                  str(round(trans_resp_mins[_r], 2)) + '\n' + str(round(trans_resp_maxs[_r], 2)))

                    # Prediction
                    ax = plt.subplot(gs1[_s-_sample_ind, 2*l_num_resp + _r-_response_ind])
                    ax.imshow(np.squeeze(pred_responses[_s, :, :, _r]),
                              vmin=pred_resp_mins[_r], vmax=pred_resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (internal_window_radius*2 != responses.shape[1]):
                        buf = (responses.shape[1] - internal_window_radius*2)/2
                        rect = patches.Rectangle((buf, buf), internal_window_radius*2,
                                                 internal_window_radius*2, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                    if (_s == _sample_ind):
                        plt.title('Predicted\nResponse ' + str(_r) + '\n' +
                                  str(round(pred_resp_mins[_r], 2)) + '\n' + str(round(pred_resp_maxs[_r], 2)))

                    # Transformed Prediction
                    ax = plt.subplot(gs1[_s-_sample_ind, 3*l_num_resp + _r-_response_ind])
                    ax.imshow(np.squeeze(invtrans_pred_responses[_s, :, :, _r]),
                              vmin=invtrans_pred_resp_mins[_r], vmax=invtrans_pred_resp_maxs[_r])
                    plt.xticks([])
                    plt.yticks([])

                    if (_s == _sample_ind):
                        plt.title('Inverse transform\nPredicted\nResponse ' + str(_r) + '\n' +
                                  str(round(invtrans_pred_resp_mins[_r], 2)) + '\n' + str(round(invtrans_pred_resp_maxs[_r], 2)))

                    if (internal_window_radius*2 != responses.shape[1]):
                        buf = (responses.shape[1] - internal_window_radius*2)/2
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


def generate_eval_report(cnn, report_name, features, responses, weights, fold_assignments, verification_fold, feature_transform, response_transform, data_config):

    # this will get more formal later on, just plugged something in
    # to break things out for now
    model_architecture = True
    input_examples_and_transformation = True
    training_history = True
    example_predictions = True

    prediction_histogram_comp = True

    spatial_error_concentration = True

    visual_stitching_artifact_check = True
    quant_stitching_artificat_check = True

    n_examples = 10

    assert os.path.isdir(os.path.dirname(report_name)), 'Invalid report location'

    if (report_name.split('.')[-1] != 'pdf'):
        report_name = report_name + '.pdf'

    with PdfPages(report_name) as pdf:

        # TODO: find max allowable number of lines, and carry over to the next page if violated
        if (model_architecture):
            fig = plot_model_summary_as_fig(cnn.model)
            pdf.savefig(fig)

        if (input_examples_and_transformation):
            figs = plot_input_examples_and_transforms(features[:n_examples, ...].copy(),
                                                      responses[:n_examples, ...].copy(),
                                                      weights.copy(),
                                                      feature_transform,
                                                      response_transform,
                                                      data_config.feature_nodata_value,
                                                      data_config.response_nodata_value)
            for fig in figs:
                pdf.savefig(fig)

        if (training_history):
            fig = plot_history(cnn.history)
            pdf.savefig(fig)

        if (example_predictions):
            figs = plot_predictions(responses[:n_examples, ...].copy(),
                                    weights.copy(),
                                    cnn.predict(feature_transform.transform(features[:n_examples, ...])),
                                    response_transform,
                                    data_config.response_nodata_value,
                                    data_config.internal_window_radius)
            for fig in figs:
                pdf.savefig(fig)

        # TODO: convert to lined graphs (borrow code from watershed work)
        if (prediction_histogram_comp):
            figs = plot_prediction_histograms(responses.copy(),
                                              weights.copy(),
                                              cnn.predict(feature_transform.transform(features)),
                                              fold_assignments,
                                              verification_fold,
                                              response_transform,
                                              data_config.response_nodata_value)
            for fig in figs:
                pdf.savefig(fig)

        if (spatial_error_concentration):
            figs = spatial_error(responses.copy(),
                                 response_transform.inverse_transform(
                                     cnn.predict(feature_transform.transform(features))),
                                 weights.copy(),
                                 data_config.response_nodata_value)
            for fig in figs:
                pdf.savefig(fig)
        # if (visual_stitching_artifact_check):
        # if (quant_stitching_artificat_check):
