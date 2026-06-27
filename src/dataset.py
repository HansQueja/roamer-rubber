import numpy as np
import albumentations as A
import os
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

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


class RubberLeafDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform  = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image  = np.array(sample['image'].convert('RGB'))
        label  = sample['label']
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label


def load_dataloaders(config: dict, hf_token: str = None):
    seed  = config['data']['seed']
    ds    = load_dataset(config['data']['dataset'],
                         num_cores = os.cpu_count(),
                         token=hf_token or True)
    data  = ds['train']
    class_names = data.features['label'].names

    split_70_30 = data.train_test_split(
        test_size=0.30, stratify_by_column='label', seed=seed)
    split_50_50 = split_70_30['test'].train_test_split(
        test_size=0.50, stratify_by_column='label', seed=seed)

    train_ds = split_70_30['train']
    val_ds   = split_50_50['train']
    test_ds  = split_50_50['test']

    bs  = config['data']['batch_size']
    nw  = config['data']['num_workers']

    train_loader = DataLoader(
        RubberLeafDataset(train_ds, get_transforms(config, 'train')),
        batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(
        RubberLeafDataset(val_ds, get_transforms(config, 'eval')),
        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(
        RubberLeafDataset(test_ds, get_transforms(config, 'eval')),
        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, class_names