import torch
import torch.nn as nn

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


def build_model(config: dict) -> nn.Module:
    name        = config['model']['name']
    num_classes = config['model']['num_classes']

    if name == 'BaselineCNN':
        return BaselineCNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. "
                         f"Add it to src/model.py")