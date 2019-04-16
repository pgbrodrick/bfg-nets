

def plot_weights(weights, ax, weight_mins, weight_maxs):
    ax.imshow(weights, vmin=weight_mins[0], vmax=weight_maxs[0], cmap='Greys_r')
