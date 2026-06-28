import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, get_device, ensure_dirs
from src.dataset import load_dataloaders
from src.model import build_model
from src.evaluate import run_evaluation
from src.explainability import run_xai

if __name__ == '__main__':
    config = load_config('configs/baseline_cnn.yaml')
    set_seed(config['data']['seed'])
    device = get_device()
    ensure_dirs(config['paths']['outputs'])

    _, val_loader, test_loader, \
    _, _, test_ds, class_names = load_dataloaders(config, device)

    model = build_model(config).to(device)

    # Evaluation must run first to get predictions
    all_preds, all_labels = run_evaluation(
        model, test_loader, class_names, device, config
    )

    # Sanity check — prints every layer in features with its index
    for i, layer in enumerate(model.features):
        print(i, layer)

    # Get the target layer for Grad-CAM
    if config['model']['name'] == 'BaselineCNN':
        target_layers = [model.conv3]
    elif config['model']['name'] == 'EnhancedCNN':
        target_layers = [model.features[-3]]  # last Conv2d before BN+ReLU
    elif config['model']['name'] == 'DeepEnhancedCNN':
        target_layers = [model.features[-3]]  # Conv2d(128, 256) — index 8
    elif config['model']['name'] == 'LeafNet':
        target_layers = [model.stage3[-2]]    # ResidualBlock's last conv
    elif config['model']['name'] == 'MobileNetEdge':
        target_layers = [model.model.features[-1]]
    else:
        raise ValueError(f"No target layer defined for {config['model']['name']}")

    run_xai(model, target_layers, test_ds,
            all_preds, all_labels, class_names,
            device, config)