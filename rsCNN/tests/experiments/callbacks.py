from rsCNN.configuration import configs
from rsCNN.experiments import callbacks


def test_get_model_callbacks_with_default_config_and_empty_history():
    config = configs.create_config_template('unet')
    assert callbacks.get_model_callbacks(config, dict())
