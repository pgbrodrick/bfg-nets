import tensorflow


def assert_gpu_available():
    session = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))
    device_types = [device.device_type for device in session.list_devices()]
    assert 'GPU' in device_types or 'XLA_GPU' in device_types, 'GPU not available in devices: {}'.format(device_types)
