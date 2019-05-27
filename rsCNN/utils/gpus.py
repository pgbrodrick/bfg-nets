import tensorflow


_GPU_TYPES = ['GPU', 'XLA_GPU']


def assert_gpu_available() -> None:
    """Asserts that a GPU is available for model training or application.

    Returns:
        None
    """
    session = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))
    device_types = [device.device_type for device in session.list_devices()]
    assert any([type_ in device_types for type_ in _GPU_TYPES]), 'GPU not available in devices: {}'.format(device_types)


def get_count_available_gpus() -> int:
    """Gets a count of GPUs available for model training or application.

    Returns:
        Number of GPUs available for model training or application.
    """
    session = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))
    device_types = [device.device_type for device in session.list_devices()]
    return len([type_ for type_ in device_types if type_ in _GPU_TYPES])
