import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np


def plot_history(history):
    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)

    # Epoch times and delays
    ax = axes.ravel()[0]
    epoch_time = [(finish - start).seconds for start, finish in zip(history['epoch_start'], history['epoch_finish'])]
    epoch_delay = [(start - finish).seconds for start, finish
                   in zip(history['epoch_start'][1:], history['epoch_finish'][:-1])]
    ax.plot(epoch_time, c='black', label='Epoch time')
    ax.plot(epoch_delay, '--', c='blue', label='Epoch delay')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Seconds')
    ax.legend()

    # Epoch times different view
    ax = axes.ravel()[1]
    dts = [epoch.strftime('%d %H:%M') for epoch in history['epoch_finish']]
    ax.hist(dts)
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylabel('Epochs completed')

    # Loss
    ax = axes.ravel()[2]
    ax.plot(history['loss'][-160:], c='black', label='Training loss')
    if 'val_loss' in history:
        ax.plot(history['val_loss'][-160:], '--', c='blue', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Learning rate
    ax = axes.ravel()[3]
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
     plt.text(0,0,model_summary_string,**{'fontsize':8,'fontfamily':'monospace'})
     plt.axis('off')
     plt.suptitle('CNN Summary')


def plot_input_examples_and_transforms(features, responses, feature_transform, response_transform, feature_nodata_value, response_nodata_value):

    features[features == feature_nodata_value] = np.nan
    responses[responses == response_nodata_value] = np.nan

    trans_features = feature_transform.transform(features)
    trans_responses = response_transform.transform(responses)

    weights = responses[...,-1]
    responses = responses[...,:-1]


    fig_list = []
    # NOTE - this is not meant to be a universal config setup, which would be annoyingly hard.  
    # This can always be exanded, but gives a reasonable amount of flexibility to start, 
    # while showing the full range of things we should actually need to see.  
    # Starting with the single response assumption.
    max_features_per_page = 3
    max_responses_per_page = 1

    max_samples_per_page = min(10,features.shape[0])
    max_pages = 8


    _feature_ind = 0
    _response_ind = 0
    _sample_ind = 0


    feat_mins = np.nanpercentile(features.reshape((-1,features.shape[-1])),0,axis=0)
    feat_maxs = np.nanpercentile(features.reshape((-1,features.shape[-1])),100,axis=0)

    resp_mins = np.nanpercentile(responses.reshape((-1,responses.shape[-1])),0,axis=0)
    resp_maxs = np.nanpercentile(responses.reshape((-1,responses.shape[-1])),100,axis=0)

    trans_feat_mins = [-2 for x in range(len(feat_mins))]
    trans_feat_maxs = [2 for x in range(len(feat_mins))]

    trans_resp_mins = [-2 for x in range(len(feat_mins))]
    trans_resp_maxs = [2 for x in range(len(feat_mins))]

    while _sample_ind < features.shape[0]:
      l_num_samp = min(max_samples_per_page, features.shape[0]-_sample_ind)
      print('sample outer block ' + str(_sample_ind))

      while _feature_ind < features.shape[-1]:
        l_num_feat = min(max_features_per_page, features.shape[-1]-_feature_ind)
        l_num_resp = min(max_responses_per_page, responses.shape[-1]-_response_ind)

        fig = plt.figure(figsize=((30*(max_features_per_page + max_responses_per_page)*2 + 1) / ((max_features_per_page + max_responses_per_page)*2 + 1 + max_samples_per_page),
                                  30*max_samples_per_page/ ((max_features_per_page + max_responses_per_page)*2 + 1 + max_samples_per_page)))
        gs1 = gridspec.GridSpec(l_num_samp,l_num_feat*2+l_num_resp*2+1)

        for _s in range(_sample_ind, _sample_ind + l_num_samp):
            for _f in range(_feature_ind, _feature_ind + l_num_feat):
                ax = plt.subplot(gs1[_s-_sample_ind,_f-_feature_ind])
                ax.imshow(np.squeeze(features[_s,:,:,_f]),vmin=feat_mins[_f],vmax=feat_maxs[_f])
                plt.xticks([])
                plt.yticks([])

                if (_f == _feature_ind):
                    plt.ylabel('Sample ' + str(_s))
                    print((_s,np.min(features[_s,:,:,_f]),np.max(features[_s,:,:,_f])))
                if (_s == _sample_ind):
                    plt.title('Feature ' + str(_f) + '\n' + str(round(feat_mins[_f],2)) + '\n' + str(round(feat_maxs[_f],2)) )

                ax = plt.subplot(gs1[_s-_sample_ind,l_num_feat + _f-_feature_ind])
                ax.imshow(np.squeeze(trans_features[_s,:,:,_f]),vmin=trans_feat_mins[_f],vmax=trans_feat_maxs[_f])
                plt.xticks([])
                plt.yticks([])

                if (_s == _sample_ind):
                    plt.title('Transformed\nFeature ' + str(_f) + '\n' + str(round(trans_feat_mins[_f],2)) + '\n' + str(round(trans_feat_maxs[_f],2)) )


            for _r in range(_response_ind, _response_ind + l_num_resp):
                ax = plt.subplot(gs1[_s-_sample_ind,2*l_num_feat + _r-_response_ind])
                ax.imshow(np.squeeze(responses[_s,:,:,_r]),vmin=resp_mins[_r],vmax=resp_maxs[_r])
                plt.xticks([])
                plt.yticks([])

                if (_s == _sample_ind):
                    plt.title('Response ' + str(_r) + '\n' + str(round(resp_mins[_r],2)) + '\n' + str(round(resp_maxs[_r],2)) )

                ax = plt.subplot(gs1[_s-_sample_ind,2*l_num_feat + l_num_resp + _r-_response_ind])
                ax.imshow(np.squeeze(trans_responses[_s,:,:,_r]),vmin=trans_resp_mins[_r],vmax=trans_resp_maxs[_r])
                plt.xticks([])
                plt.yticks([])

                if (_s == _sample_ind):
                    plt.title('Transformed\nResponse ' + str(_r) + '\n' + str(round(trans_resp_mins[_r],2)) + '\n' + str(round(trans_resp_maxs[_r],2)) )
            ax = plt.subplot(gs1[_s-_sample_ind,-1])
            ax.imshow(np.squeeze(weights[_s,:,:]),vmin=0,vmax=1,cmap='Greys_r')
            plt.xticks([])
            plt.yticks([])

            if (_s == _sample_ind):
                plt.title('Weights')


        fig_list.append(fig) 
        _feature_ind += max_features_per_page

        if (len(fig_list) > max_pages):
          break
      _sample_ind += max_samples_per_page
      if (len(fig_list) > max_pages):
        break

    return fig_list


def generate_eval_report(cnn, report_name, features, responses, feature_transform, response_transform, data_config):

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








    assert os.path.isdir(os.path.dirname(report_name)), 'Invalid report location'
   
    if (report_name.split('.')[-1] != 'pdf'):
      report_name = report_name + '.pdf'

    with PdfPages(report_name) as pdf: 

        #TODO: find max allowable number of lines, and carry over to the next page if violated
        if (model_architecture):
            plot_model_summary_as_fig(cnn.model)
            pdf.savefig()
        
        
        if (input_examples_and_transformation):
            figs = plot_input_examples_and_transforms(features[:10,...],
                                                     responses[:10,...],
                                                     feature_transform,
                                                     response_transform,
                                                     data_config.feature_nodata_value,
                                                     data_config.response_nodata_value)
            for fig in figs:
                pdf.savefig(fig)

        # TODO: handle subfigure spacing better
        if (training_history):
            fig = plot_history(cnn.history)
            pdf.savefig(fig)



        #if (example_predictions):
        #if (prediction_histogram_comp):
        #if (spatial_error_concentration):
        #if (visual_stitching_artifact_check):
        #if (quant_stitching_artificat_check):





   






