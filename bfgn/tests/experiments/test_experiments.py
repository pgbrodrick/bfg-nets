import os

import keras.backend as K
import pytest

from bfgn.configuration import configs
from bfgn.experiments import experiments, histories, models


@pytest.fixture()
def config_new(tmp_path) -> configs.Config:
    config = configs.create_config_from_file(
        os.path.join(os.path.dirname(__file__), "test_config.yaml")
    )
    config.data_build.dir_out = str(tmp_path)
    config.model_training.dir_out = str(tmp_path)
    return config


def test_experiment_initialized_without_existing(config_new) -> None:
    experiment = experiments.Experiment(config_new)
    assert type(experiment) is experiments.Experiment


def test_experiment_initialized_with_existing(config_new) -> None:
    experiment_a = experiments.Experiment(config_new)
    experiment_b = experiments.Experiment(config_new)
    assert type(experiment_b) is experiments.Experiment


def test_experiment_asserts_error_on_initialization_with_conflicting_config(
    config_new
) -> None:
    experiment_a = experiments.Experiment(config_new)
    config = config_new
    config.data_build.max_samples += 1
    with pytest.raises(AssertionError):
        experiments.Experiment(config)


def test_experiment_build_or_load_model_builds_new_model_successfully(
    config_new
) -> None:
    experiment = experiments.Experiment(config_new)
    experiment.build_or_load_model(data_container=None, num_features=1)
    assert experiment.loaded_existing_model is False
    assert experiment.loaded_existing_history is False
    assert experiment.is_model_trained is False


def test_experiment_build_or_load_model_loads_existing_model_successfully(
    config_new
) -> None:
    experiment_a = experiments.Experiment(config_new)
    experiment_a.build_or_load_model(data_container=None, num_features=1)
    experiment_b = experiments.Experiment(config_new)
    experiment_b.build_or_load_model(data_container=None, num_features=1)
    assert experiment_b.loaded_existing_model is True
    assert experiment_b.loaded_existing_history is True


def test_experiment__build_new_model_builds_saves_sets_attrs(config_new) -> None:
    experiment = experiments.Experiment(config_new)
    experiment._build_new_model((3, 3, 2))
    assert experiment.model
    assert os.path.exists(experiment.filepath_model)
    assert experiment.loaded_existing_model is False


def test_experiment__build_new_history_builds_saves_sets_attrs(config_new) -> None:
    experiment = experiments.Experiment(config_new)
    experiment._build_new_history()
    assert experiment.history
    assert os.path.exists(experiment.filepath_history)
    assert experiment.loaded_existing_history is False
    assert experiment.is_model_trained is False


def test_experiment__load_existing_model_loads_sets_attrs(config_new) -> None:
    input_shape = (3, 3, 2)
    experiment_a = experiments.Experiment(config_new)
    experiment_a._build_new_model(input_shape)
    experiment_b = experiments.Experiment(config_new)
    experiment_b._load_existing_model(input_shape)
    assert experiment_b.model
    assert experiment_b.loaded_existing_model is True


def test_experiment__load_existing_model_wrong_input_shape_raises(config_new) -> None:
    experiment_a = experiments.Experiment(config_new)
    experiment_a._build_new_model((3, 3, 2))
    experiment_b = experiments.Experiment(config_new)
    with pytest.raises(AssertionError):
        experiment_b._load_existing_model((4, 4, 3))


def test_experiment__load_existing_history_loads_sets_attrs_with_val_loss(
    config_new
) -> None:
    experiment = experiments.Experiment(config_new)
    expected_history = {
        "val_loss": [1, 1, 0, 1],
        "lr": [1, 1, 0, 1],
        experiments.KEY_HISTORY_IS_MODEL_TRAINED: True,
    }
    histories.save_history(expected_history, experiment.filepath_history)
    experiment._build_new_model((3, 3, 2))
    experiment._load_existing_history()
    assert experiment.history == expected_history
    assert experiment._init_epoch == 3
    assert K.get_value(experiment.model.optimizer.lr) == 0
    assert experiment.loaded_existing_history is True
    assert experiment.is_model_trained is True


def test_experiment__load_existing_history_loads_sets_attrs_no_val_loss(
    config_new
) -> None:
    experiment = experiments.Experiment(config_new)
    expected_history = {
        "lr": [1, 1, 0, 1],
        experiments.KEY_HISTORY_IS_MODEL_TRAINED: False,
    }
    histories.save_history(expected_history, experiment.filepath_history)
    experiment._build_new_model((3, 3, 2))
    experiment._load_existing_history()
    assert experiment.history == expected_history
    assert experiment._init_epoch == 4
    assert K.get_value(experiment.model.optimizer.lr) == 1
    assert experiment.loaded_existing_history is True
    assert experiment.is_model_trained is False


def test_experiment_fit_model_with_sequences_raises_on_existing_model_no_resume(
    config_new
) -> None:
    experiment_a = experiments.Experiment(config_new)
    experiment_a.build_or_load_model(num_features=1)
    experiment_b = experiments.Experiment(config_new)
    experiment_b.build_or_load_model(num_features=1)
    with pytest.raises(AssertionError):
        experiment_b.fit_model_with_sequences(None, None, resume_training=False)


def test_experiment_fit_model_with_data_container_does_everything(
    config_new, monkeypatch
) -> None:
    experiment = experiments.Experiment(config_new)
    experiment.build_or_load_model(num_features=1)

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


def test_experiment_fit_model_with_sequences_does_everything(
    config_new, monkeypatch
) -> None:
    experiment = experiments.Experiment(config_new)
    experiment.build_or_load_model(num_features=1)

    class MockReturn:
        history = {"lr": [3, 2, 1], "val_loss": [3, 2, 1]}

    def mock_history(*args, **kwargs):
        return MockReturn

    def mock_model_save(*args, **kwargs):
        return True

    monkeypatch.setattr(experiment.model, "fit_generator", mock_history)
    monkeypatch.setattr(models, "save_model", mock_model_save)

    experiment.fit_model_with_sequences(None)
    assert experiment.is_model_trained is True
    assert experiment.history[experiments.KEY_HISTORY_IS_MODEL_TRAINED] is True
    assert histories.load_history(experiment.filepath_history) == experiment.history


def test_load_experiment_from_directory_returns_experiment(config_new) -> None:
    experiment_a = experiments.Experiment(config_new)
    experiment_b = experiments.load_experiment_from_directory(
        config_new.model_training.dir_out
    )
    assert type(experiment_b) is experiments.Experiment
