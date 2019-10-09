import numpy as np
import pytest

from bfgn.reporting.visualizations import samples


@pytest.fixture()
def mock_sampled(tmp_path) -> object:
    class MockDataBuild:
        window_radius = 2
        loss_window_radius = 1

    class MockConfig:
        data_build = MockDataBuild

    class MockDataSeq:
        nan_replacement_value = 1

    class MockSampled:
        data_sequence = MockDataSeq
        data_sequence_label = "test"
        model = None
        is_model_trained = True
        config = MockConfig
        num_samples = 3
        num_features = 4
        num_responses = 5
        has_features_transform = False
        has_responses_transform = False
        raw_features = np.ones((3, 8, 8, 4))
        raw_features_range = np.ones((4, 2))
        trans_features = np.ones((3, 8, 8, 4))
        trans_features_range = np.ones((4, 2))
        raw_responses = np.ones((3, 8, 8, 5))
        raw_responses_range = np.ones((5, 2))
        trans_responses = np.ones((3, 8, 8, 5))
        trans_responses_range = np.ones((5, 2))
        raw_predictions = np.ones((3, 8, 8, 5))
        raw_predictions_range = np.ones((5, 2))
        trans_predictions = np.ones((3, 8, 8, 5))
        trans_predictions_range = np.ones((5, 2))
        weights = np.ones((3, 8, 8))
        weights_range = np.ones((1, 2))
        feature_band_types = ["R"] * 4
        response_band_types = ["R"] * 5

    return MockSampled


def test__plot_samples_passes_classification_excessive_maxes(mock_sampled) -> None:
    max_pages = 20
    max_samples = 10
    max_features = 10
    max_responses = 10
    sample_type = samples.LABEL_CLASSIFICATION
    samples._plot_samples(mock_sampled, max_pages, max_samples, max_features, max_responses, sample_type)


def test__plot_samples_passes_regression_excessive_maxes(mock_sampled) -> None:
    max_pages = 20
    max_samples = 10
    max_features = 10
    max_responses = 10
    sample_type = samples.LABEL_REGRESSION
    samples._plot_samples(mock_sampled, max_pages, max_samples, max_features, max_responses, sample_type)


def test__plot_samples_passes_limited_features(mock_sampled) -> None:
    max_pages = 20
    max_samples = 10
    max_features = 3
    max_responses = 10
    sample_type = samples.LABEL_REGRESSION
    samples._plot_samples(mock_sampled, max_pages, max_samples, max_features, max_responses, sample_type)


def test__plot_samples_passes_limited_responses(mock_sampled) -> None:
    max_pages = 20
    max_samples = 10
    max_features = 10
    max_responses = 3
    sample_type = samples.LABEL_REGRESSION
    samples._plot_samples(mock_sampled, max_pages, max_samples, max_features, max_responses, sample_type)


def test__plot_samples_passes_limited_pages(mock_sampled) -> None:
    max_pages = 1
    max_samples = 2
    max_features = 10
    max_responses = 10
    sample_type = samples.LABEL_REGRESSION
    samples._plot_samples(mock_sampled, max_pages, max_samples, max_features, max_responses, sample_type)


def test__plot_samples_passes_multiple_pages(mock_sampled) -> None:
    max_pages = 10
    max_samples = 1
    max_features = 10
    max_responses = 10
    sample_type = samples.LABEL_REGRESSION
    samples._plot_samples(mock_sampled, max_pages, max_samples, max_features, max_responses, sample_type)


def test__plot_classification_sample_passes_no_plots() -> None:
    class MockSampled:
        raw_predictions = True

    idx_sample = 0
    num_features = 3
    num_responses = 3
    sample_axes = iter([None] * 50)
    samples._plot_classification_sample(MockSampled, idx_sample, num_features, num_responses, sample_axes)


def test__plot_regression_sample_passes_no_plots() -> None:
    class MockSampled:
        raw_predictions = True

    idx_sample = 0
    num_features = 3
    num_responses = 3
    sample_axes = iter([None] * 50)
    samples._plot_regression_sample(MockSampled, idx_sample, num_features, num_responses, sample_axes)
