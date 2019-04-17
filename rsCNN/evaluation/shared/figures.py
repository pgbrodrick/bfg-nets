import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


_FIGSIZE_CONSTANT = 30


def get_figure_and_grid(nrows, ncols):
    width = _FIGSIZE_CONSTANT * ncols / (ncols + nrows)
    height = _FIGSIZE_CONSTANT * nrows / (ncols + nrows)
    fig = plt.figure(figsize=(width, height))
    grid = gridspec.GridSpec(nrows, ncols)
    return fig, grid
