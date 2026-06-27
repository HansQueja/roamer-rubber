# src/dataset.py
import os
import platform
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(config: dict, mode: str):
    aug      = config['augmentation']
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class LocalRubberDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, segmenter=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform
        self.segmenter   = segmenter

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image    = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not read image at: {img_path}\n"
                f"Check that the path in your split.csv is correct."
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        # YOLO segmentation runs before augmentation so augmentation
        # operates on the already-cleaned leaf image
        if self.segmenter is not None:
            from src.segmentation import segment_leaf_yolo
            image = segment_leaf_yolo(self.segmenter, image)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


# ── Split CSV loader ──────────────────────────────────────────────────────────

def _load_split_from_csv(config: dict):
    """
    Reads split.csv from the dataset root.

    Expected columns:
        file_path  — relative path from dataset root (Windows backslash ok)
        label      — integer class index (unused, we re-derive from class_name)
        class_name — string class name e.g. 'Anthracnose'
        split      — 'train', 'val', or 'test'
    """
    COL_FILENAME = "file_path"
    COL_LABEL    = "class_name"
    COL_SPLIT    = "split"

    dataset_root = config['data']['dataset']
    csv_path     = os.path.join(dataset_root, "split.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"split.csv not found at: {csv_path}\n"
            f"Make sure it exists inside your dataset root folder."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded split.csv: {len(df)} total rows")
    print(f"Split distribution:\n{df[COL_SPLIT].value_counts().to_string()}\n")

    # Derive class index from class_name string — sorted for reproducibility
    # This ignores the pre-existing 'label' integer column intentionally
    # so class ordering is consistent regardless of what the CSV assigned
    class_names  = sorted(df[COL_LABEL].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def _extract(split_name):
        subset = df[df[COL_SPLIT] == split_name].reset_index(drop=True)
        if len(subset) == 0:
            raise ValueError(
                f"No rows found for split='{split_name}' in split.csv.\n"
                f"Expected values in the '{COL_SPLIT}' column: train, val, test."
            )
        paths = [
            os.path.join(
                dataset_root,
                # Replace Windows backslashes with forward slashes
                # before joining — critical when CSV was made on Windows
                # but is being read on Linux (Colab)
                row[COL_FILENAME].replace("\\", "/")
            )
            for _, row in subset.iterrows()
        ]
        labels = [class_to_idx[row[COL_LABEL]] for _, row in subset.iterrows()]
        return paths, labels

    train_paths, train_labels = _extract("train")
    val_paths,   val_labels   = _extract("val")
    test_paths,  test_labels  = _extract("test")

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print(f"Classes ({len(class_names)}): {class_names}\n")

    return (train_paths, train_labels,
            val_paths,   val_labels,
            test_paths,  test_labels,
            class_names)


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_num_workers(config: dict) -> int:
    """Force num_workers=0 on Windows — spawn-based multiprocessing
    breaks with YOLO inside __getitem__ and with HuggingFace datasets."""
    if platform.system() == 'Windows':
        return 0
    return config['data'].get('num_workers', 2)


def load_dataloaders(config: dict, device=None, segmenter=None):
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels,
     class_names) = _load_split_from_csv(config)

    train_ds = LocalRubberDataset(
        train_paths, train_labels,
        transform=get_transforms(config, 'train'),
        segmenter=segmenter
    )
    val_ds = LocalRubberDataset(
        val_paths, val_labels,
        transform=get_transforms(config, 'eval'),
        segmenter=segmenter
    )
    test_ds = LocalRubberDataset(
        test_paths, test_labels,
        transform=get_transforms(config, 'eval'),
        segmenter=segmenter
    )

    bs  = config['data']['batch_size']
    nw  = get_num_workers(config)
    pin = (device is not None and device.type == 'cuda')

    # YOLO inside __getitem__ cannot be pickled across worker processes
    if segmenter is not None and nw > 0:
        print("Warning: num_workers forced to 0 — "
              "YOLO segmenter cannot be pickled across worker processes.")
        nw = 0

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    return (train_loader, val_loader, test_loader,
            train_ds, val_ds, test_ds, class_names)