import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend('Agg')  # Needed for remote server plotting


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


def print_model_summary(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary_string = "\n".join(stringlist)

    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
    plt.text(0, 0, model_summary_string, **{'fontsize': 8, 'fontfamily': 'monospace'})
    plt.axis('off')
    plt.suptitle('CNN Summary')

    return fig



def adjust_axis(lax):
  for sp in lax.spines:
    lax.spines[sp].set_color('white')
    lax.spines[sp].set_linewidth(2)
  lax.set_xticks([])
  lax.set_yticks([])



def visualize_feature_progression(data_sequence, model, full_vertical=False, max_filters=10, item_index=0):

    # Grab out the necessary data to pass through the network
    feature, response, weight = data_sequence.__getitem__(0)
    features = features[0]
    responses = responses[0]
    responses, weights = responses[..., :-1], responses[..., -1]

    feature = feature[item_index,...]
    feature = feature.reshape((1,feature.shape[0],feature.shape[1],feature.shape[2]))

    response = response[item_index,...]
    response = response.reshape((1,response.shape[0],response.shape[1],response.shape[2]))

    weight = weight[item_index,...]
    weight = weight.reshape((1,weight.shape[0],weight.shape[1],weight.shape[2]))
    

    # Run through the model and grab any Conv2D layer (other layers could also be grabbed as desired)
    pred_set = []
    layer_names = []
    pred_set.append(feature)
    layer_names.append('Feature')
    for _l in range(0,len(model.layers)):
        if (isinstance(model.layers[_l] , keras.layers.convolutional.Conv2D)):
            im_model = keras.models.Model(inputs = model.layers[0].output, outputs = model.layers[_l].output)
            pred_set.append(im_model.predict(feature))
            layer_names.append(model.layers[_l].name)
    pred_set.append(response)
    layer_names.append('Response')
    
    # Calculate the per-filter standard deviation, enables plots to preferentially show more interesting layers
    pred_std = []
    for _l in range(len(pred_set)):
        pred_std.append([np.std(np.squeeze(pred_set[_l][...,x])) for x in range(0,pred_set[_l].shape[-1])])
    

    # Get spacing things worked out and the figure initialized
    step_size = 1 / float(len(pred_set)+1)
    if (full_vertical):
        h_space_fraction = 0.05
    else:
        h_space_fraction = 0.3

    image_size = min(step_size * (1-h_space_fraction), 1 / max_filters * (1-h_space_fraction))
    h_space_size = step_size*h_space_fraction

    fig = plt.figure(figsize=(max(max_filters,len(pred_set)), max(max_filters,len(pred_set))))
    

    # Step through each layer in the network
    for _l in range(0,len(pred_set)):
        # Step through each filter, up to the max
        for _iii in range(0,min(pred_set[_l].shape[-1],max_filters)):

            if (full_vertical):
                ip = [(_l+0.5)*step_size,_iii*image_size*(1+h_space_fraction)]
            else:
                ip = [(_l+0.5)*step_size + _iii*h_space_size/5.,_iii*image_size*0.2]
    
            # Get the indices sorted by filter std, as a proxy for interest
            ordered_pred_std = np.argsort(pred_std[_l])[::-1]
            # prep the image
            tp = np.squeeze(pred_set[_l][item_index,:,:,ordered_pred_std[_iii]])

            #Plot!
            ax = fig.add_axes([ip[0],ip[1],image_size,image_size],zorder=max_filters+1-_iii)
            plt.imshow(tp,vmin=np.nanpercentile(tp,5),vmax=np.nanpercentile(tp,95))
            adjust_axis(ax)
            if (_iii == 0):
                plt.xlabel(layer_names[_l])
 
    
    return [fig]





































