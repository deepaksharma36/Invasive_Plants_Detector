import yaml
from easydict import EasyDict as edict

cfg =  edict()

def yfile_to_cfg(file_path):
    with open(file_path) as fp:
        cfg_file = edict(yaml.load(fp))
    return cfg_file
