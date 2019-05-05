from rsCNN import configs
from rsCNN.networks import architectures
from rsCNN.utils import DIR_TEMPLATES


if __name__ == '__main__':
    architecture_names = architectures.get_available_architectures()
    for architecture_name in architecture_names:
        configs.create_config_template(architecture_name, DIR_TEMPLATES, architecture_name + '.yaml')
