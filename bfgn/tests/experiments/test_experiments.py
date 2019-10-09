import os

import keras.backend as K
import numpy as np
import pytest

from bfgn.configuration import configs
from bfgn.experiments import experiments, histories, models

DEFAULT_NUM_FEATURES = 1
DEFAULT_HISTORY = {"val_loss": [4, 3, 2, 1], "lr": [9, 8, 7, 8], experiments.KEY_HISTORY_IS_MODEL_TRAINED: False}


@pytest.fixture()
def config_new(tmp_path) -> configs.Config:
    config = configs.create_config_from_file(os.path.join(os.path.dirname(__file__), "test_config.yaml"))
    config.data_build.dir_out = str(tmp_path)
    config.model_training.dir_out = str(tmp_path)
    return config


@pytest.fixture()
def config_existing(tmp_path) -> configs.Config:
    config = configs.create_config_from_file(os.path.join(os.path.dirname(__file__), "test_config.yaml"))
    config.data_build.dir_out = str(tmp_path)
    config.model_training.dir_out = str(tmp_path)
    experiment = experiments.Experiment(config)
    experiment.build_or_load_model(num_features=DEFAULT_NUM_FEATURES)
    return config


def test_experiment_initialized_with_and_without_existing_experiment(config_new) -> None:
    experiment_a = experiments.Experiment(config_new)
    assert type(experiment_a) is experiments.Experiment
    experiment_b = experiments.Experiment(config_new)
    assert type(experiment_b) is experiments.Experiment


def test_experiment_asserts_error_on_initialization_with_conflicting_config(config_existing) -> None:
    config = config_existing
    config.data_build.max_samples += 1
    with pytest.raises(AssertionError):
        experiments.Experiment(config)


def test_experiment_build_or_load_model_sets_new_model_history_attrs(config_new) -> None:
    experiment = experiments.Experiment(config_new)
    experiment.build_or_load_model(num_features=DEFAULT_NUM_FEATURES)
    assert experiment.model
    assert os.path.exists(experiment.filepath_model)
    assert experiment.loaded_existing_model is False
    assert experiment.history
    assert os.path.exists(experiment.filepath_history)
    assert experiment.loaded_existing_history is False
    assert experiment.is_model_trained is False


def test_experiment_build_or_load_model_sets_existing_model_history_attrs(config_existing) -> None:
    experiment = experiments.Experiment(config_existing)
    experiment.build_or_load_model(num_features=DEFAULT_NUM_FEATURES)
    assert experiment.model
    assert os.path.exists(experiment.filepath_model)
    assert experiment.loaded_existing_model is True
    assert experiment.history
    assert os.path.exists(experiment.filepath_history)
    assert experiment.loaded_existing_history is True
    assert experiment.is_model_trained is False


def test_experiment_build_or_load_model_existing_wrong_input_shape_raises(config_existing) -> None:
    experiment = experiments.Experiment(config_existing)
    with pytest.raises(AssertionError):
        experiment.build_or_load_model(num_features=DEFAULT_NUM_FEATURES + 1)


def test_experiment__load_existing_history_loads_sets_attrs_with_val_loss(config_new) -> None:
    experiment = experiments.Experiment(config_new)
    history = DEFAULT_HISTORY.copy()
    history[experiments.KEY_HISTORY_IS_MODEL_TRAINED] = True
    histories.save_history(history, experiment.filepath_history)
    window_radius = config_new.data_build.window_radius
    input_shape = (window_radius * 2, window_radius * 2, DEFAULT_NUM_FEATURES)
    experiment._build_new_model(input_shape)
    experiment._load_existing_history()
    idx_min = int(np.argmin(history["val_loss"]))
    assert experiment.history == history
    assert experiment._init_epoch == idx_min + 1
    assert K.get_value(experiment.model.optimizer.lr) == history["lr"][idx_min]
    assert experiment.loaded_existing_history is True
    assert experiment.is_model_trained is True


def test_experiment__load_existing_history_loads_sets_attrs_no_val_loss(config_new) -> None:
    experiment = experiments.Experiment(config_new)
    history = DEFAULT_HISTORY.copy()
    history.pop("val_loss")
    histories.save_history(history, experiment.filepath_history)
    window_radius = config_new.data_build.window_radius
    input_shape = (window_radius * 2, window_radius * 2, DEFAULT_NUM_FEATURES)
    experiment._build_new_model(input_shape)
    experiment._load_existing_history()
    idx_min = len(history["lr"]) - 1
    assert experiment.history == history
    assert experiment._init_epoch == idx_min + 1
    assert K.get_value(experiment.model.optimizer.lr) == history["lr"][idx_min]
    assert experiment.loaded_existing_history is True
    assert experiment.is_model_trained is False


def test_experiment_fit_model_with_sequences_raises_on_existing_model_no_resume(config_existing) -> None:
    experiment = experiments.Experiment(config_existing)
    experiment.loaded_existing_model = True
    with pytest.raises(AssertionError):
        experiment.fit_model_with_sequences(None, resume_training=False)


def test_experiment_fit_model_with_data_container_and_sequences(config_new, monkeypatch) -> None:
    experiment = experiments.Experiment(config_new)
    experiment.build_or_load_model(num_features=DEFAULT_NUM_FEATURES)

    class MockDataContainer:
        training_sequence = None
        validation_sequence = None

    class MockReturn:
        history = {"lr": [3, 2, 1], "val_loss": [3, 2, 1]}

    def mock_history(*args, **kwargs):
        return MockReturn

    def mock_model_save(*args, **kwargs):
        return True

    monkeypatch.setattr(experiment.model, "fit_generator", mock_history)
    monkeypatch.setattr(models, "save_model", mock_model_save)

    experiment.fit_model_with_data_container(MockDataContainer)
    assert experiment.is_model_trained is True
    assert experiment.history[experiments.KEY_HISTORY_IS_MODEL_TRAINED] is True
    assert histories.load_history(experiment.filepath_history) == experiment.history

    experiment.is_model_trained = False

    experiment.fit_model_with_sequences(None, resume_training=True)
    assert experiment.is_model_trained is True
    assert experiment.history[experiments.KEY_HISTORY_IS_MODEL_TRAINED] is True
    assert histories.load_history(experiment.filepath_history) == experiment.history


def test_load_experiment_from_directory_returns_experiment(config_existing) -> None:
    experiment = experiments.load_experiment_from_directory(config_existing.model_training.dir_out)
    assert type(experiment) is experiments.Experiment
