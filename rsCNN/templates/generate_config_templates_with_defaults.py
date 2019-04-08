from rsCNN.networks import network_configs
from rsCNN.utils import DIR_TEMPLATES


if __name__ == '__main__':
    value_required = 'REQUIRED'
    for architecture in ('change_detection', 'regress_net', 'residual_net', 'residual_unet', 'unet'):
        network_config = network_configs.create_network_config(
            architecture=architecture, model_name=value_required, inshape=(0, 0, 0),
            n_classes=0, loss_metric=value_required, output_activation=value_required,
        )
        network_configs.save_network_config(network_config, dir_config=DIR_TEMPLATES, filename=architecture + '.ini')
