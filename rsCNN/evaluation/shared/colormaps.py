COLORMAP_CATEGORICAL = 'tab20'
_CATEGORICAL_CLASSES_MAX = 20  # From tab20, it'd be 10 if we were using tab10

COLORMAP_ERROR = 'RdBu'
COLORMAP_METRICS = 'Blues'
COLORMAP_WEIGHTS = 'Greys_r'


def check_is_categorical_colormap_repeated(num_classes):
    return num_classes > _CATEGORICAL_CLASSES_MAX
