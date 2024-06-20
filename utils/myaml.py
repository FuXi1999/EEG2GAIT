import yaml

from dotmap import DotMap

def load_config(config_file_path):
    with open(config_file_path) as file:
        config = yaml.safe_load(file)

    return DotMap(config)