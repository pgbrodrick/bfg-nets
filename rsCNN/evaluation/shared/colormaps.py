from matplotlib import colors


_COLOR_CORRECT = (59, 117, 175)
_COLOR_INCORRECT = (238, 134, 54)
COLORMAP_ERROR = colors.ListedColormap([_COLOR_INCORRECT, _COLOR_CORRECT])

COLORMAP_CATEGORICAL = 'tab20'
_CATEGORICAL_CLASSES_MAX = 20  # From tab20, it'd be 10 if we were using tab10

COLORMAP_METRICS = 'Blues'
COLORMAP_WEIGHTS = 'Greys_r'


def check_is_categorical_colormap_repeated(num_classes):
    return num_classes > _CATEGORICAL_CLASSES_MAX
