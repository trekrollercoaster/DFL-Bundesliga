import torch
import json
import os

BASE_PATH = os.getcwd()
ARG_PATH = os.path.join(BASE_PATH, "config.json")
with open(ARG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
