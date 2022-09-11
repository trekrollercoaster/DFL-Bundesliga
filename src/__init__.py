import torch
import yaml
import os

BASE_PATH = os.getcwd()
config_path = os.path.join(BASE_PATH, "config.yml")
CONFIG = yaml.load(open(config_path, 'r', encoding="utf-8"), Loader=yaml.FullLoader)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
