# scripts/generate_xai_comparison.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model   import build_model
from src.dataset import load_dataloaders
from src.utils   import load_config, get_device

# ── Preprocessing — must match training exactly ───────────────────────────────
_preprocess = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ── Target layer registry — keep in sync with benchmark.py ───────────────────

def get_target_layers(model, arch_name: str):
    """
    Returns the Grad-CAM target layers for each architecture.
    Must match the layer used in scripts/xai.py and src/explainability.py.
    """
    if arch_name == 'BaselineCNN':
        return [model.conv3]
    elif arch_name == 'EnhancedCNN':
        return [model.features[-3]]
    elif arch_name == 'DeepEnhancedCNN':
        return [model.features[-3]]    # Conv2d(128, 256) — last conv before GAP
    elif arch_name == 'LeafNet':
        return [model.stage3[-2]]      # last ResidualBlock in stage3
    elif arch_name == 'MobileNetEdge':
        return [model.model.features[-1]]
    else:
        raise ValueError(
            f"No Grad-CAM target layer defined for '{arch_name}'.\n"
            f"Add it to get_target_layers() in generate_xai_comparison.py"
        )


# ── Model loader — takes arch name directly, no config file needed ────────────

def load_model_for_xai(arch_name: str,
                        checkpoint_path: str,
                        num_classes: int,
                        device: torch.device):
    """
    Build model from architecture name (not config file) and load checkpoint.
    Returns (model, target_layers).
    """
    config = {
        'model': {'name': arch_name, 'num_classes': num_classes}
    }
    model = build_model(config).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device))
    model.eval()

    target_layers = get_target_layers(model, arch_name)
    return model, target_layers


# ── Grad-CAM overlay ──────────────────────────────────────────────────────────

def get_gradcam_overlay(model, target_layers, raw_img_rgb: np.ndarray,
                         target_class: int, device: torch.device) -> np.ndarray:
    """
    Compute Grad-CAM heatmap overlay on a raw RGB numpy image.
    raw_img_rgb: H x W x 3 uint8, RGB order.
    """
    input_tensor = _preprocess(image=raw_img_rgb)['image'] \
                               .unsqueeze(0).to(device)
    targets      = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Normalise to [0,1] float for overlay — do NOT re-normalise
    rgb_float = cv2.resize(raw_img_rgb, (224, 224)) \
                          .astype(np.float32) / 255.0
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)


# ── Single-image comparison ───────────────────────────────────────────────────

def generate_comparison(img_idx: int,
                         test_ds,
                         class_names: list,
                         models_to_compare: list,
                         device: torch.device,
                         output_dir: str = 'outputs/xai'):
    """
    Generate a side-by-side Grad-CAM comparison figure for one test image.

    models_to_compare: list of dicts, each with keys:
        'arch'  — architecture name string (matches build_model registry)
        'ckpt'  — path to checkpoint .pth file
        'label' — display label for the figure

    Example:
        models_to_compare = [
            {'arch': 'DeepEnhancedCNN',
             'ckpt': 'checkpoints/best_deepenhanced_cnn.pth',
             'label': 'DeepEnhancedCNN + YOLO'},
            {'arch': 'MobileNetEdge',
             'ckpt': 'checkpoints/best_mobilenet.pth',
             'label': 'MobileNetEdge + YOLO'},
        ]
    """
    # ── Load image from test_ds (already points to segmented dataset) ─────────
    img_path       = test_ds.image_paths[img_idx]
    true_label_idx = test_ds.labels[img_idx]
    true_class     = class_names[true_label_idx]

    raw_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    print(f"Image  : {img_path}")
    print(f"True   : {true_class} (idx {true_label_idx})")

    # ── Build figure — original + one column per model ────────────────────────
    n_models = len(models_to_compare)
    fig, axes = plt.subplots(2, 1 + n_models,
                             figsize=(5 * (1 + n_models), 10))
    fig.suptitle(
        f"Grad-CAM Comparison — True class: {true_class}\n"
        f"Test index: {img_idx} | {os.path.basename(img_path)}",
        fontsize=13
    )

    # Column 0 — original segmented image
    axes[0, 0].imshow(raw_img)
    axes[0, 0].set_title("Segmented Input", fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')   # empty bottom-left cell

    # Columns 1..N — one per model
    for col, mcfg in enumerate(models_to_compare, start=1):
        arch  = mcfg['arch']
        ckpt  = mcfg['ckpt']
        label = mcfg['label']

        if not os.path.exists(ckpt):
            print(f"  ⚠️  Skipping '{label}' — checkpoint not found: {ckpt}")
            axes[0, col].set_title(f"{label}\n(checkpoint missing)", fontsize=9)
            axes[0, col].axis('off')
            axes[1, col].axis('off')
            continue

        print(f"  Generating CAM for: {label}...")
        model, target_layers = load_model_for_xai(
            arch, ckpt, len(class_names), device)

        # Get model's prediction for this image
        with torch.no_grad():
            tensor = _preprocess(image=raw_img)['image'] \
                                .unsqueeze(0).to(device)
            probs   = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx])

        correct = pred_idx == true_label_idx
        status  = "✓" if correct else "✗"

        # CAM towards the model's prediction (not forced to true label)
        # This shows what the model actually looked at for its answer
        overlay = get_gradcam_overlay(model, target_layers,
                                       raw_img, pred_idx, device)

        # Top row: original + prediction info
        axes[0, col].imshow(raw_img)
        axes[0, col].set_title(
            f"{label}\nPred: {pred_class} {status} ({confidence:.1%})",
            fontsize=9,
            color='green' if correct else 'red'
        )
        axes[0, col].axis('off')

        # Bottom row: Grad-CAM heatmap
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(
            f"CAM → '{pred_class}'",
            fontsize=9
        )
        axes[1, col].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir,
                            f"xai_comparison_idx{img_idx}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved → {out_path}")
    return out_path


# ── Batch comparison — one image per class ────────────────────────────────────

def generate_per_class_comparison(test_ds,
                                   class_names: list,
                                   models_to_compare: list,
                                   device: torch.device,
                                   output_dir: str = 'outputs/xai',
                                   prefer_correct: bool = True):
    """
    Generate one comparison figure per class, picking a representative
    test image for each.

    prefer_correct: if True, pick an image all models got right.
                    if False, pick one where models disagree (more interesting).
    """
    # First pass — collect predictions from all models for all test images
    print("Collecting predictions for all models...")

    all_model_preds = {}
    for mcfg in models_to_compare:
        arch  = mcfg['arch']
        ckpt  = mcfg['ckpt']
        label = mcfg['label']

        if not os.path.exists(ckpt):
            print(f"  ⚠️  Skipping '{label}' — checkpoint not found")
            continue

        model, _ = load_model_for_xai(arch, ckpt,
                                        len(class_names), device)
        preds = []
        with torch.no_grad():
            for path in test_ds.image_paths:
                img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                t    = _preprocess(image=img)['image'].unsqueeze(0).to(device)
                pred = int(torch.argmax(model(t), dim=1).item())
                preds.append(pred)
        all_model_preds[label] = preds
        print(f"  Done: {label}")

    # Second pass — pick one image per class
    for target_class_idx, class_name in enumerate(class_names):
        # Candidate indices: test images of this class
        candidates = [
            i for i, lbl in enumerate(test_ds.labels)
            if lbl == target_class_idx
        ]
        if not candidates:
            print(f"  No test images for class: {class_name}")
            continue

        if prefer_correct:
            # Prefer images all models classified correctly
            agreed_correct = [
                i for i in candidates
                if all(all_model_preds[m][i] == target_class_idx
                       for m in all_model_preds)
            ]
            chosen = agreed_correct[0] if agreed_correct else candidates[0]
        else:
            # Prefer images where models disagree
            disagreed = [
                i for i in candidates
                if len(set(all_model_preds[m][i]
                           for m in all_model_preds)) > 1
            ]
            chosen = disagreed[0] if disagreed else candidates[0]

        print(f"\nClass '{class_name}': using test index {chosen}")
        generate_comparison(chosen, test_ds, class_names,
                            models_to_compare, device,
                            output_dir=output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    config      = load_config('configs/baseline_cnn.yaml')
    device      = get_device()
    _, _, _, _, _, test_ds, class_names = load_dataloaders(config, device)

    # ── Define which models to compare ────────────────────────────────────────
    # Add or remove entries here — arch names must match build_model registry
    MODELS = [
        {
            'arch' : 'DeepEnhancedCNN',
            'ckpt' : 'checkpoints/best_deepenhanced_cnn.pth',
            'label': 'DeepEnhancedCNN + YOLO',
        },
        {
            'arch' : 'MobileNetEdge',
            'ckpt' : 'checkpoints/best_mobilenet.pth',
            'label': 'MobileNetEdge + YOLO',
        },
    ]

    # ── Option A: single specific image ───────────────────────────────────────
    #TARGET_INDEX = 1
    #generate_comparison(TARGET_INDEX, test_ds, class_names,
    #                    MODELS, device)

    # ── Option B: one representative image per class ───────────────────────────
    # Uncomment to generate the full per-class grid for the paper
    generate_per_class_comparison(
        test_ds, class_names, MODELS, device,
        prefer_correct=True
    )