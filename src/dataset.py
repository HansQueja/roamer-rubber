import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.datasets as dset

def get_transforms(config: dict, mode: str):
    """
    mode: 'train' or 'eval'
    """
    aug = config['augmentation']
    img_size = config['data']['image_size']

    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size),
                                scale=aug['random_resized_crop']['scale']),
            A.HorizontalFlip(p=aug['horizontal_flip']),
            A.VerticalFlip(p=aug['vertical_flip']),
            A.Rotate(limit=aug['rotate_limit'], p=0.5),
            A.RandomBrightnessContrast(p=aug['brightness_contrast']),
            A.HueSaturationValue(
                hue_shift_limit=aug['hue_saturation']['hue_shift_limit'],
                sat_shift_limit=aug['hue_saturation']['sat_shift_limit'],
                val_shift_limit=aug['hue_saturation']['val_shift_limit'],
                p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


class LocalRubberDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read directly from local disk using fast OpenCV
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label


def load_dataloaders(config: dict):
    seed = config['data']['seed']
    # Ensure this matches the key you put in your YAML (e.g., '/content/Unified_Dataset')
    dataset_path = config['data']['dataset_path'] 
    
    # We use ImageFolder strictly as a fast way to map the directory structure
    print(f"Scanning local directory: {dataset_path}...")
    full_dataset = dset.ImageFolder(dataset_path)
    class_names = full_dataset.classes
    
    # Extract raw lists of paths and labels
    all_paths = [item[0] for item in full_dataset.imgs]
    all_labels = full_dataset.targets

    # Perform the Stratified 70/30 Split
    train_paths, test_val_paths, train_labels, test_val_labels = train_test_split(
        all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=seed
    )
    
    # Perform the Stratified 15/15 Split (Half of the 30)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_val_paths, test_val_labels, test_size=0.50, stratify=test_val_labels, random_state=seed
    )

    # Initialize our custom PyTorch datasets
    train_ds = LocalRubberDataset(train_paths, train_labels, get_transforms(config, 'train'))
    val_ds   = LocalRubberDataset(val_paths, val_labels, get_transforms(config, 'eval'))
    test_ds  = LocalRubberDataset(test_paths, test_labels, get_transforms(config, 'eval'))

    bs = config['data']['batch_size']
    nw = config['data']['num_workers']

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, class_names