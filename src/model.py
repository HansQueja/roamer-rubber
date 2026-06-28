import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# --- 1. Your Original Baseline (For the IEEE Ablation Study) ---
class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(p=0.3)
        self.gap   = nn.AdaptiveAvgPool2d(1)

        self.fc1   = nn.Linear(64, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

# --- 2. An Enhanced Custom CNN (Deeper feature extraction) ---
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Extract significantly more features (up to 256)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    

class DeepEnhancedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: 56 → 28 — double conv adds depth without losing resolution
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # second conv, no pool
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4: 28 → 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)   # 256-dim after flatten

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),               # increased from 0.4
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),               # increased from 0.2
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- 3. The Edge Robotics Champion (Pre-trained) ---
class MobileNetEdge(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Load the pre-trained weights to instantly boost accuracy
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Freeze early layers to speed up training and prevent overfitting (optional but recommended)
        for param in self.model.features[:10].parameters():
            param.requires_grad = False
            
        # Replace the final classification head for our 6 specific classes
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

# --- 4. The Model Builder ---
def build_model(config: dict) -> nn.Module:
    name        = config['model']['name']
    num_classes = config['model']['num_classes']

    if name == 'BaselineCNN':
        return BaselineCNN(num_classes)
    elif name == 'EnhancedCNN':
        return EnhancedCNN(num_classes)
    elif name == 'DeepEnhancedCNN':
        return DeepEnhancedCNN(num_classes)
    elif name == 'MobileNetEdge':
        return MobileNetEdge(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Add it to src/model.py")