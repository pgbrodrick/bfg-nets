from matplotlib import gridspec
import matplotlib.pyplot as plt


_FIGSIZE_CONSTANT = 1.5


def get_figure_and_grid(nrows, ncols):
    width = _FIGSIZE_CONSTANT * ncols
    height = _FIGSIZE_CONSTANT * nrows
    fig = plt.figure(figsize=(width, height))
    grid = gridspec.GridSpec(nrows, ncols)
    return fig, grid
