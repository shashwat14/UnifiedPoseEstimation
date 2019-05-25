import yaml

class Struct:

    def __init__(self, **params):
        self.__dict__.update(params)

with open ('../cfg/cfg.yaml', 'r') as f:
    params = yaml.safe_load(f)

parameters = Struct(**params)