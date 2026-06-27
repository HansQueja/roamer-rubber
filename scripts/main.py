import sys
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
sys.path.append('..')

from src.utils import load_config, set_seed, get_device, ensure_dirs
from src.dataset import load_dataloaders
from src.model import build_model
from scripts.train import train

torch.backends.cudnn.benchmark = True

config = load_config('configs/baseline_cnn.yaml')
set_seed(config['data']['seed'])
device = get_device()
ensure_dirs(config['paths']['outputs'], 'checkpoints')

# ── Load YOLO segmenter if enabled in config ──────────────────────────────────
segmenter = None
seg_cfg   = config.get('segmentation', {})
if seg_cfg.get('enabled', False):
    from src.segmentation import load_yolo_segmenter
    segmenter = load_yolo_segmenter(seg_cfg['model_path'])
    print(f"YOLO segmenter loaded: {seg_cfg['model_path']}\n")
else:
    print("Segmentation disabled — training on raw images.\n")

# ── Load data ─────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader, \
train_ds, val_ds, test_ds, class_names = load_dataloaders(
    config, device=device, segmenter=segmenter
)

# ── Class weights ─────────────────────────────────────────────────────────────
print("Calculating class weights to balance dataset...")
all_train_labels  = train_ds.labels
class_weights_np  = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_train_labels),
    y=all_train_labels
)
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float)
print(f"Computed weights for {len(class_names)} classes: {class_weights_np}\n")

# ── Train ─────────────────────────────────────────────────────────────────────
model = build_model(config).to(device)
train(model, train_loader, val_loader, config, device,
      class_weights=class_weights_tensor)