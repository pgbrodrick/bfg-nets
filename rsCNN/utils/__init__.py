import os

import tensorflow


DIR_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_TEMPLATES = os.path.join(DIR_PROJECT_ROOT, 'templates')

_GPU_TYPES = ['GPU', 'XLA_GPU']


def assert_gpu_available():
    session = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))
    device_types = [device.device_type for device in session.list_devices()]
    assert any([type_ in device_types for type_ in _GPU_TYPES]), 'GPU not available in devices: {}'.format(device_types)
