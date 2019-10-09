import matplotlib.pyplot as plt
import numpy as np

from bfgn.reporting.visualizations import subplots


def test_plot_raw_features_passes() -> None:
    class MockSampled:
        raw_features = np.ones((3, 4, 4, 1))
        raw_features_range = np.array([0, 1]).reshape((1, 2))

    fig, ax = plt.subplots()
    subplots.plot_raw_features(MockSampled, 0, 0, ax, True, True)


def test_plot_transformed_features_passes() -> None:
    class MockDataSeq:
        nan_replacement_value = 1

    class MockSampled:
        trans_features = np.ones((3, 4, 4, 1))
        trans_features_range = np.array([0, 1]).reshape((1, 2))
        data_sequence = MockDataSeq

    fig, ax = plt.subplots()
    subplots.plot_transformed_features(MockSampled, 0, 0, ax, True, True)


def test_plot_raw_responses_passes() -> None:
    class MockSampled:
        raw_responses = np.ones((3, 4, 4, 1))
        raw_responses_range = np.array([0, 1]).reshape((1, 2))

    fig, ax = plt.subplots()
    subplots.plot_raw_responses(MockSampled, 0, 0, ax, True, True)


def test_plot_transformed_responses_passes() -> None:
    class MockDataSeq:
        nan_replacement_value = 1

    class MockSampled:
        trans_responses = np.ones((3, 4, 4, 1))
        trans_responses_range = np.array([0, 1]).reshape((1, 2))
        data_sequence = MockDataSeq

    fig, ax = plt.subplots()
    subplots.plot_transformed_responses(MockSampled, 0, 0, ax, True, True)


def test_plot_categorical_responses_passes() -> None:
    class MockSampled:
        raw_responses = np.ones((3, 4, 4, 1))
        num_responses = 2

    fig, ax = plt.subplots()
    subplots.plot_categorical_responses(MockSampled, 0, 0, ax, True, True)


def test_plot_raw_predictions_passes() -> None:
    class MockDataBuild:
        window_radius = 2
        loss_window_radius = 1

    class MockConfig:
        data_build = MockDataBuild

    class MockSampled:
        raw_predictions = np.ones((3, 4, 4, 1))
        raw_predictions_range = np.array([0, 1]).reshape((1, 2))
        config = MockConfig

    fig, ax = plt.subplots()
    subplots.plot_raw_predictions(MockSampled, 0, 0, ax, True, True)


def test_plot_transformed_predictions_passes() -> None:
    class MockDataBuild:
        window_radius = 2
        loss_window_radius = 1

    class MockConfig:
        data_build = MockDataBuild

    class MockDataSeq:
        nan_replacement_value = 1

    class MockSampled:
        trans_predictions = np.ones((3, 4, 4, 1))
        trans_predictions_range = np.array([0, 1]).reshape((1, 2))
        config = MockConfig
        data_sequence = MockDataSeq

    fig, ax = plt.subplots()
    subplots.plot_transformed_predictions(MockSampled, 0, 0, ax, True, True)


def test__plot_sample_attribute_returns_early_no_axis() -> None:
    ax = None
    subplots._plot_sample_attribute(None, None, None, None, None, None, None)


def test__plot_sample_attribute_returns_early_no_attribute() -> None:
    class MockSampled:
        raw_responses = None

    fig, ax = plt.subplots()
    subplots._plot_sample_attribute(MockSampled, None, None, "raw_responses", ax, None, None)


def test__plot_sample_attribute_passes_categorical() -> None:
    class MockSampled:
        raw_responses = np.ones((3, 4, 4, 1))
        num_responses = 2

    fig, ax = plt.subplots()
    subplots._plot_sample_attribute(MockSampled, 0, 0, "categorical_responses", ax, True, True)


def test__plot_sample_attribute_passes_continuous() -> None:
    class MockSampled:
        raw_responses = np.ones((3, 4, 4, 1))
        raw_responses_range = np.array([0, 1]).reshape((1, 2))

    fig, ax = plt.subplots()
    subplots._plot_sample_attribute(MockSampled, 0, 0, "raw_responses", ax, True, True)


def test_plot_classification_predictions_max_likelihood_returns_early_no_axis() -> None:
    subplots.plot_classification_predictions_max_likelihood(None, None, None, None, None)


def test_plot_classification_predictions_max_likelihood_returns_early_no_data() -> None:
    class MockSampled:
        weights = None

    subplots.plot_classification_predictions_max_likelihood(MockSampled, None, None, None, None)


def test_plot_classification_predictions_max_likelihood_passes() -> None:
    class MockSampled:
        num_responses = 4
        raw_predictions = np.ones((3, 4, 4, 4))


def test_plot_weights_returns_early_no_axis() -> None:
    subplots.plot_weights(None, None, None, None, None)


def test_plot_weights_returns_early_no_data() -> None:
    class MockSampled:
        weights = None

    subplots.plot_weights(MockSampled, None, None, None, None)


def test_plot_weights_passes() -> None:
    class MockSampled:
        weights = np.ones((3, 4, 4))
        weights_range = np.ones((1, 2))

    fig, ax = plt.subplots()
    subplots.plot_weights(MockSampled, 0, ax, True, True)


def test_plot_binary_error_regression_returns_early_no_axis() -> None:
    subplots.plot_binary_error_classification(None, None, None, None, None)


def test_plot_binary_error_regression_returns_early_no_data() -> None:
    class MockSampled:
        raw_responses = None
        raw_predictions = None

    subplots.plot_binary_error_classification(MockSampled, None, None, None, None)


def test_plot_binary_error_regression_passes() -> None:
    class MockDataBuild:
        window_radius = 2
        loss_window_radius = 1

    class MockConfig:
        data_build = MockDataBuild

    class MockSampled:
        raw_predictions = np.ones((3, 4, 4, 5))
        raw_responses = np.ones((3, 4, 4, 5))
        config = MockConfig

    fig, ax = plt.subplots()
    subplots.plot_binary_error_classification(MockSampled, 0, ax, True, True)


def test_plot_raw_error_regression_returns_early_no_axis() -> None:
    subplots.plot_raw_error_regression(None, None, None, None, None, None)


def test_plot_raw_error_regression_returns_early_no_data() -> None:
    class MockSampled:
        raw_responses = None
        raw_predictions = None

    subplots.plot_raw_error_regression(MockSampled, None, None, None, None, None)


def test_plot_raw_error_regression_passes() -> None:
    class MockDataBuild:
        window_radius = 2
        loss_window_radius = 2

    class MockConfig:
        data_build = MockDataBuild

    class MockSampled:
        raw_predictions = np.ones((3, 4, 4, 5))
        raw_responses = np.ones((3, 4, 4, 5))
        config = MockConfig

    fig, ax = plt.subplots()
    subplots.plot_raw_error_regression(MockSampled, 0, 0, ax, True, True)


def test_plot_transformed_error_regression_returns_early_no_axis() -> None:
    subplots.plot_transformed_error_regression(None, None, None, None, None, None)


def test_plot_transformed_error_regression_returns_early_no_data() -> None:
    class MockSampled:
        trans_responses = None
        trans_predictions = None

    subplots.plot_transformed_error_regression(MockSampled, None, None, None, None, None)


def test_plot_transformed_error_regression_passes() -> None:
    class MockDataBuild:
        window_radius = 2
        loss_window_radius = 1

    class MockConfig:
        data_build = MockDataBuild

    class MockSampled:
        raw_predictions = np.ones((3, 4, 4, 5))
        raw_responses = np.ones((3, 4, 4, 5))
        trans_predictions = np.ones((3, 4, 4, 5))
        trans_responses = np.ones((3, 4, 4, 5))
        config = MockConfig

    fig, ax = plt.subplots()
    subplots.plot_transformed_error_regression(MockSampled, 0, 0, ax, True, True)


def test_add_internal_window_to_subplot_passes() -> None:
    class MockDataBuild:
        window_radius = 2
        loss_window_radius = 1

    class MockConfig:
        data_build = MockDataBuild

    class MockSampled:
        raw_predictions = np.ones((3, 4, 4, 5))
        raw_responses = np.ones((3, 4, 4, 5))
        config = MockConfig

    fig, ax = plt.subplots()
    subplots.add_internal_window_to_subplot(MockSampled, ax)


def test__format_number_int_valid() -> None:
    assert subplots._format_number(1) == "1"


def test__format_number_float_valid() -> None:
    assert subplots._format_number(1.2345678) == "1.23"
