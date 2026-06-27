import torch
import random
import numpy as np
import yaml
import os
from huggingface_hub import login

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return device

def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def huggingface_login(token):
    HF_TOKEN = token

    print("Logging into Hugging Face...")
    login(token=HF_TOKEN)