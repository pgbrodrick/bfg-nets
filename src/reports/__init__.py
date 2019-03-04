import os

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(predictions, targets, band_order, dir_out=None):
    """
    Creates a PDF of predictions and targets
    :param predictions: NN predictions in [n_images, image_dim, image_dim, bands] array
    :param targets: NN targets in [n_images, image_dim, image_dim, bands] array
    :param band_order: which bands from the preds/targs to use as RGB bands
    :param dir_out: output directory if not directly shown
    :return:
    """
    # TODO:  we may need to generalize band_order or how bands are handled, I had a specific use-case
    # Expect predictions and targets to be lists of arrays, multiple predictions and targets can come from NNs
    for idx_target, (raw_targs, raw_preds) in enumerate(zip(targets, predictions)):
        # Create images that have RGB dimensions
        shape = list(raw_targs.shape)[:-1] + [3]  # Enforce three bands
        targs = np.full(shape, np.nan)
        shape = list(raw_preds.shape)[:-1] + [3]  # Enforce three bands
        preds = np.full(shape, np.nan)
        # Fill RGB images with the specified bands, e.g., band_order (None, 1, 0) means that the RGB image will have no
        # red band, a green band from the first dimension, and a blue band from the zeroth dimension.
        for idx_plot, idx_orig in enumerate(band_order):
            if idx_orig is None:
                continue
            targs[:, :, :, idx_plot] = raw_targs[:, :, :, idx_orig]
            preds[:, :, :, idx_plot] = raw_preds[:, :, :, idx_orig]
        # Generate PDF page
        path_out = None
        if dir_out:
            path_out = os.path.join(dir_out, 'target_{}.pdf'.format(idx_target))
        _plot_predictions_page(preds, targs, path_out)


def _plot_predictions_page(predictions, targets, path_out=None):
    NROWS_PAGE = 8
    NCOLS = 4
    # Create output directory, if necessary,
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))
    # Calculate number of images total, number of rows, number of columns
    nrows_total = int(np.ceil(targets.shape[0] / 2))
    num_pages = int(np.ceil(nrows_total / NROWS_PAGE))
    # Transform image data to avoid warnings
    with np.errstate(invalid='ignore'):
        targets[targets > 1] = 1
        targets[targets < 0] = 0
        predictions[predictions > 1] = 1
        predictions[predictions < 0] = 0
    # Generate PDF with figures
    with PdfPages(path_out) as pdf:
        # Create a figure on every page
        for _ in range(num_pages):
            fig, axes = plt.subplots(figsize=(2*NCOLS, 2*NROWS_PAGE), nrows=NROWS_PAGE, ncols=NCOLS,
                                     gridspec_kw={'wspace': 0.025, 'hspace': 0.050})
            # Plot pairs of images in the figure until the page is full
            for idx_axis, axis in enumerate(axes.ravel()):
                axis.axis('off')
                if idx_axis % 2 == 0:  # First plot in each pair
                    axis.imshow(targets[0, :, :, :])
                    targets = targets[1:, :, :, :]
                else:  # Second plot in each pair
                    axis.imshow(predictions[0, :, :, :])
                    predictions = predictions[1:, :, :, :]
                    if predictions.shape[0] == 0:
                        break  # No targets or predictions left to show
            # Remove any extra axes
            for idx_axis in range(1 + idx_axis, len(axes.ravel())):
                axes.ravel()[idx_axis].remove()
            if path_out:
                pdf.savefig(fig)
            else:
                fig.show()
    return
