import sys
sys.path.append('..')

from src.utils import load_config, set_seed, get_device, ensure_dirs, huggingface_login
from src.dataset import load_dataloaders
from src.model import build_model
from scripts.train import train

# Login via huggingface for datasets
huggingface_login()

config = load_config('configs/baseline_cnn.yaml')
set_seed(config['data']['seed'])
device = get_device()
ensure_dirs(config['paths']['outputs'], 'checkpoints')

train_loader, val_loader, test_loader, \
train_ds, val_ds, test_ds, class_names = load_dataloaders(config)

model = build_model(config).to(device)
train(model, train_loader, val_loader, config, device)