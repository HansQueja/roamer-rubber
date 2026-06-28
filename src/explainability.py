import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ── Preprocessing (same as eval_transforms, no augmentation) ──────────────────
_preprocess = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def _load_raw_image(test_ds, idx: int) -> np.ndarray:
    """
    Load the original unnormalised RGB image from disk.
    Works with LocalRubberDataset which stores file paths.
    """
    img_path = test_ds.image_paths[idx]
    image    = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_gradcam_overlay(model, target_layers, raw_image_np,
                        target_class, device):
    """
    Compute Grad-CAM overlay for a single raw RGB numpy image.
    target_class: integer class index to explain.
    """
    input_tensor = _preprocess(image=raw_image_np)['image'] \
                               .unsqueeze(0).to(device)
    targets      = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)[0]

    rgb_float = cv2.resize(raw_image_np, (224, 224)) \
                          .astype(np.float32) / 255.0
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)


def collect_fp_fn_records(model, test_ds, all_preds, all_labels,
                           class_names, device):
    records = []
    model.eval()

    with torch.no_grad():
        for idx in range(len(test_ds)):
            true_c = int(all_labels[idx])
            pred_c = int(all_preds[idx])

            if true_c == pred_c:
                continue

            raw_img = _load_raw_image(test_ds, idx)   # ← fixed
            tensor  = _preprocess(image=raw_img)['image'] \
                                 .unsqueeze(0).to(device)
            probs   = torch.softmax(model(tensor), dim=1) \
                           .cpu().numpy()[0]

            records.append({
                'idx'       : idx,
                'true_c'    : true_c,
                'pred_c'    : pred_c,
                'true_name' : class_names[true_c],
                'pred_name' : class_names[pred_c],
                'confidence': float(probs[pred_c]),
                'true_conf' : float(probs[true_c]),
                'raw_image' : raw_img,
            })

    return records


def print_fp_fn_summary(records, class_names):
    print(f"Total misclassified: {len(records)}\n")
    print(f"{'Class':<20} {'FN (missed)':>12} {'FP (false alarm)':>17}")
    print("─" * 52)
    for c, cname in enumerate(class_names):
        fn = sum(1 for r in records if r['true_c'] == c)
        fp = sum(1 for r in records if r['pred_c'] == c)
        print(f"{cname:<20} {fn:>12} {fp:>17}")


def plot_correct_predictions(model, target_layers, test_ds,
                              all_preds, all_labels, class_names,
                              device, output_dir):
    n   = len(class_names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    fig.suptitle(
        "Grad-CAM — Correct Predictions\n"
        "Top: Original | Bottom: CAM Overlay",
        fontsize=13
    )

    for c, cname in enumerate(class_names):
        match = next(
            (i for i in range(len(test_ds))
             if all_labels[i] == c and all_preds[i] == c),
            None
        )
        if match is None:
            axes[0, c].set_title(f"{cname}\n(none correct)")
            axes[0, c].axis('off')
            axes[1, c].axis('off')
            continue

        raw_img = _load_raw_image(test_ds, match)   # ← fixed
        overlay = get_gradcam_overlay(model, target_layers,
                                      raw_img, c, device)

        axes[0, c].imshow(raw_img)
        axes[0, c].set_title(cname, fontsize=9)
        axes[0, c].axis('off')

        axes[1, c].imshow(overlay)
        axes[1, c].set_title(f"Predicted: {cname}", fontsize=9)
        axes[1, c].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "gradcam_correct.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Saved → {path}")


def _dual_cam_grid(model, target_layers, records, class_names,
                   device, output_dir, mode):
    """
    Shared logic for FP and FN dual-CAM grids.
    mode: 'fp' or 'fn'
    """
    assert mode in ('fp', 'fn')

    for c, cname in enumerate(class_names):
        if mode == 'fp':
            subset = [r for r in records if r['pred_c'] == c]
            title  = (f"False Positives — Predicted as '{cname}'\n"
                      f"Row 1: Original  |  "
                      f"Row 2: CAM→'{cname}' (wrong)  |  "
                      f"Row 3: CAM→True class (correct)")
            fname  = f"FP_{cname.replace(' ', '_')}.png"
        else:
            subset = [r for r in records if r['true_c'] == c]
            title  = (f"False Negatives — True '{cname}' was missed\n"
                      f"Row 1: Original  |  "
                      f"Row 2: CAM→Wrong prediction  |  "
                      f"Row 3: CAM→'{cname}' (correct)")
            fname  = f"FN_{cname.replace(' ', '_')}.png"

        if not subset:
            label = "false positives" if mode == 'fp' else "false negatives"
            print(f"\n{cname}: No {label} — skipping.")
            continue

        n   = len(subset)
        fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
        if n == 1:
            axes = axes.reshape(3, 1)
        fig.suptitle(title, fontsize=10, y=1.01)

        for col, r in enumerate(subset):
            raw_img     = r['raw_image']
            cam_wrong   = get_gradcam_overlay(model, target_layers,
                                              raw_img, r['pred_c'], device)
            cam_correct = get_gradcam_overlay(model, target_layers,
                                              raw_img, r['true_c'], device)

            # Row 0 — original
            axes[0, col].imshow(raw_img)
            axes[0, col].set_title(
                f"True: {r['true_name']}\n"
                f"Conf(wrong):{r['confidence']:.2f}  "
                f"Conf(true):{r['true_conf']:.2f}",
                fontsize=7.5
            )
            axes[0, col].axis('off')

            # Row 1 — CAM towards wrong class
            axes[1, col].imshow(cam_wrong)
            axes[1, col].set_title(
                f"Focused on:\n'{r['pred_name']}' (wrong)",
                fontsize=7.5, color='red'
            )
            axes[1, col].axis('off')

            # Row 2 — CAM towards correct class
            axes[2, col].imshow(cam_correct)
            axes[2, col].set_title(
                f"Should focus on:\n'{r['true_name']}'",
                fontsize=7.5, color='green'
            )
            axes[2, col].axis('off')

        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.show()
        print(f"Saved → {path}")


def run_xai(model, target_layers, test_ds, all_preds, all_labels,
            class_names, device, config):
    output_dir = os.path.join(config['paths']['outputs'], 'xai')
    os.makedirs(output_dir, exist_ok=True)

    # Correct predictions
    print("Grad-CAM: Correct predictions (one per class)")
    plot_correct_predictions(model, target_layers, test_ds,
                             all_preds, all_labels, class_names,
                             device, output_dir)

    # Collect all errors
    print("\nCollecting misclassification records...")
    records = collect_fp_fn_records(model, test_ds, all_preds,
                                    all_labels, class_names, device)
    print_fp_fn_summary(records, class_names)

    # FP grids
    print("\nGenerating False Positive grids...")
    _dual_cam_grid(model, target_layers, records, class_names,
                   device, output_dir, mode='fp')

    # FN grids
    print("\nGenerating False Negative grids...")
    _dual_cam_grid(model, target_layers, records, class_names,
                   device, output_dir, mode='fn')

    print(f"\nAll XAI outputs saved to: {output_dir}")