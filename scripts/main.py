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

# ── Segmentation setup ────────────────────────────────────────────────────────
seg_cfg   = config.get('segmentation', {})
segmenter = None

if seg_cfg.get('enabled', False):
    from src.segmentation import load_yolo_segmenter, presegment_dataset
    from src.dataset import _load_split_from_csv

    yolo = load_yolo_segmenter(seg_cfg['model_path'])

    # Load all paths from CSV to pre-segment
    (train_paths, _, val_paths, _,
     test_paths, _, _) = _load_split_from_csv(config)
    all_paths = train_paths + val_paths + test_paths

    seg_output_dir = seg_cfg.get('presegment_output',
                                  'Unified_Dataset_Segmented')

    print(f"Pre-segmenting {len(all_paths)} images → {seg_output_dir}")
    presegment_dataset(yolo, all_paths, seg_output_dir,
                       fill=seg_cfg.get('fill', 'white'),
                       conf_threshold=seg_cfg.get('conf_threshold', 0.25))

    # Point config to the segmented dataset for dataloaders
    # so they load from disk with no live YOLO overhead
    config['data']['dataset'] = seg_output_dir
    print(f"Training will use pre-segmented images from: {seg_output_dir}\n")
else:
    print("Segmentation disabled — training on raw images.\n")

# load_dataloaders now reads from segmented dir if enabled
# no segmenter= argument needed since images are already processed
train_loader, val_loader, test_loader, \
train_ds, val_ds, test_ds, class_names = load_dataloaders(config, device=device)

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
train(model, train_loader, val_loader, config, device)