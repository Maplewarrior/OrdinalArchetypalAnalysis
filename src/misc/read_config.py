import yaml

def load_config(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    cfg = load_config('configs/config.yaml')
    import pdb
    pdb.set_trace()