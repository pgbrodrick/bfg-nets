from bfgn.reporting.visualizations import colormaps


def test_check_is_categorical_colormap_repeated_true() -> None:
    assert colormaps.check_is_categorical_colormap_repeated(100) is True


def test_check_is_categorical_colormap_repeated_false() -> None:
    assert colormaps.check_is_categorical_colormap_repeated(1) is False
