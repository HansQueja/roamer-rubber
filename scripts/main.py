import sys
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
sys.path.append('..')

from src.utils import load_config, set_seed, get_device, ensure_dirs
from src.dataset import load_dataloaders
from src.model import build_model
from scripts.train import train

# For training optimization
torch.backends.cudnn.benchmark = True

config = load_config('configs/baseline_cnn.yaml')
set_seed(config['data']['seed'])
device = get_device()
ensure_dirs(config['paths']['outputs'], 'checkpoints')

# Load the data
train_loader, val_loader, test_loader, \
train_ds, val_ds, test_ds, class_names = load_dataloaders(config)

# --- NEW: Calculate Class Weights to handle Powdery Mildew imbalance ---
print("Calculating class weights to balance dataset...")
# Extract all labels from the training dataset
all_train_labels = train_ds.labels 
class_weights_np = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_train_labels),
    y=all_train_labels  
)

# Convert to PyTorch float tensor
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float)
print(f"Computed weights for 6 classes: {class_weights_np}\n")
# -----------------------------------------------------------------------

model = build_model(config).to(device)

# Pass the weights tensor into your updated train function
train(model, train_loader, val_loader, config, device, class_weights=class_weights_tensor)