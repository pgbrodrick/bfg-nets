from rsCNN.networks.architectures import regress_net, residual_net, residual_unet, unet


def get_architecture_creator(architecture_name):
    try:
        creator = globals()[architecture_name]
    except AttributeError:
        NotImplementedError('Invalid network type: ' + architecture_name)
    return creator
