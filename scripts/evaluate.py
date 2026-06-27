import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, get_device, ensure_dirs
from src.dataset import load_dataloaders
from src.model import build_model
from src.evaluate import run_evaluation

if __name__ == '__main__':
    config = load_config('configs/baseline_cnn.yaml')
    set_seed(config['data']['seed'])
    device = get_device()
    ensure_dirs(config['paths']['outputs'])

    _, val_loader, test_loader, \
    _, _, test_ds, class_names = load_dataloaders(config, device)

    model = build_model(config).to(device)

    all_preds, all_labels = run_evaluation(
        model, test_loader, class_names, device, config
    )