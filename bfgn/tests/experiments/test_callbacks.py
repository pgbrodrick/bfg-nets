from bfgn.configuration import configs
from bfgn.experiments import callbacks


def test_get_model_callbacks_with_default_config_and_empty_history():
    config = configs.create_config_template('unet')
    assert callbacks.get_model_callbacks(config, dict())
