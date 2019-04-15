from rsCNN.networks.architectures \
    import change_detection, dilation_net, flat_net, residual_dilation_net, residual_flat_net, residual_unet, unet


def get_architecture_creator(architecture_name):
    assert architecture_name in globals(), 'Invalid network name: {}'.format(architecture_name)
    return globals()[architecture_name]
