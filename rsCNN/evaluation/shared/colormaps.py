from matplotlib import colors


_COLOR_CORRECT = tuple([x / 255 for x in (158, 202, 225)])
_COLOR_INCORRECT = tuple([x / 255 for x in (253, 174, 107)])
COLORMAP_ERROR = colors.ListedColormap([_COLOR_INCORRECT, _COLOR_CORRECT])

COLORMAP_METRICS = 'Blues'

COLORMAP_WEIGHTS = 'Greys_r'
