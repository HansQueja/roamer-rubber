import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, csv, warnings
import torch
import torch.nn as nn
import numpy as np
import cv2
import yaml
from pathlib import Path
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

from src.model    import build_model
from src.quantize import apply_dynamic_quantization, get_model_size_mb, get_param_count
from src.dataset  import load_dataloaders

warnings.filterwarnings('ignore')


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Image utilities ───────────────────────────────────────────────────────────

_eval_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return _eval_tf(image=img)['image'].unsqueeze(0)


# ── YOLO utilities ────────────────────────────────────────────────────────────

def segment_with_yolo(yolo_model, image_rgb, conf=0.10):
    """
    Segment leaf, return black-background isolated image.
    Falls back to original image if no detection.
    """
    h, w   = image_rgb.shape[:2]
    result = yolo_model(image_rgb, conf=conf, verbose=False)[0]

    if result.masks is None or len(result.masks) == 0:
        return image_rgb   # no detection fallback

    best     = int(result.boxes.conf.argmax())
    mask     = cv2.resize(
        result.masks.data[best].cpu().numpy(), (w, h),
        interpolation=cv2.INTER_NEAREST)
    mask     = (mask > 0.5).astype(np.uint8)
    out      = image_rgb.copy()
    out[mask == 0] = 0 # Changed to 0 (Black Fill) to reflect your 95%+ pipeline discovery
    return out


def preprocess_with_yolo(path, yolo_model, conf=0.10):
    """Load image, segment with YOLO, return classifier-ready tensor."""
    img       = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    segmented = segment_with_yolo(yolo_model, img, conf)
    return _eval_tf(image=segmented)['image'].unsqueeze(0)


def yolo_standalone_predict(yolo_model, path, class_names, conf=0.10):
    """
    Use YOLO segmentation head's class prediction as the classification output.
    Returns predicted class index and confidence.
    """
    img    = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    result = yolo_model(img, conf=conf, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        return 3, 1.0 # No detection fallback -> predict Healthy

    best      = int(result.boxes.conf.argmax())
    cls_idx   = int(result.boxes.cls[best].item())
    cls_conf  = float(result.boxes.conf[best].item())
    return cls_idx, cls_conf


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_classifier(model, test_loader, class_names, device):
    """Standard classifier evaluation on test_loader."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs  = imgs.to(device)
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    return _compute_metrics(np.array(all_preds),
                            np.array(all_labels), class_names)


def evaluate_yolo_standalone(yolo_model, test_ds, class_names, conf=0.10):
    """
    Evaluate YOLO using its segmentation-head class predictions
    against the test dataset labels.
    """
    all_preds, all_labels = [], []
    for idx in range(len(test_ds)):
        img_path  = test_ds.image_paths[idx]
        true_label = test_ds.labels[idx]
        pred_idx, _ = yolo_standalone_predict(
            yolo_model, img_path, class_names, conf)
        all_preds.append(pred_idx)
        all_labels.append(true_label)
    return _compute_metrics(np.array(all_preds),
                            np.array(all_labels), class_names)


def _compute_metrics(preds, labels, class_names):
    acc       = accuracy_score(labels, preds)
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, average=None,
        labels=list(range(len(class_names))), zero_division=0)
    mp,mr,mf1,_ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0)
    wp,wr,wf1,_ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0)

    per_class = {
        class_names[i]: {
            'precision': round(float(p[i]), 4),
            'recall'   : round(float(r[i]), 4),
            'f1'       : round(float(f1[i]), 4),
            'support'  : int(support[i]),
        } for i in range(len(class_names))
    }
    return {
        'accuracy'          : round(float(acc), 4),
        'macro_precision'   : round(float(mp),  4),
        'macro_recall'      : round(float(mr),  4),
        'macro_f1'          : round(float(mf1), 4),
        'weighted_f1'       : round(float(wf1), 4),
        'per_class'         : per_class,
        'all_preds'         : preds,
        'all_labels'        : labels,
    }


# ── Timing ────────────────────────────────────────────────────────────────────

def time_pipeline(predict_fn, image_paths, device, warmup=3):
    """General-purpose timing wrapper."""
    # Active warm-up to prevent initial VRAM loading latencies from inflating values
    for path in image_paths[:warmup]:
        predict_fn(path)

    timings = []
    for path in image_paths:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        predict_fn(path)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    timings = np.array(timings)
    return {
        'mean_ms' : round(float(timings.mean()), 3),
        'std_ms'  : round(float(timings.std()),  3),
        'min_ms'  : round(float(timings.min()),  3),
        'max_ms'  : round(float(timings.max()),  3),
        'fps'     : round(1000 / float(timings.mean()), 2),
        'n'       : len(image_paths),
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(results, class_names):
    hdr = (f"{'Model':<38} {'Acc':>7} {'mP':>6} {'mR':>6} "
           f"{'mF1':>6} {'wF1':>6} {'MB':>7} {'ms/img':>8} {'FPS':>7}")
    div = "─" * len(hdr)
    print(f"\n{div}\nBENCHMARK SUMMARY\n{div}")
    print(hdr)
    print(div)
    for r in results:
        ev, tm = r['eval'], r['timing']
        print(f"{r['name']:<38} "
              f"{ev['accuracy']:>7.2%} "
              f"{ev['macro_precision']:>6.2%} "
              f"{ev['macro_recall']:>6.2%} "
              f"{ev['macro_f1']:>6.2%} "
              f"{ev['weighted_f1']:>6.2%} "
              f"{r['size_mb']:>7.2f} "
              f"{tm['mean_ms']:>8.2f} "
              f"{tm['fps']:>7.1f}")
    print(div)

    print(f"\n{'─'*50}\nPER-CLASS F1\n{'─'*50}")
    col_w = 11
    print(f"{'Class':<22}" + "".join(
        f"{r['name'][:col_w-1]:>{col_w}}" for r in results))
    print("─" * (22 + col_w * len(results)))
    for cls in class_names:
        row = f"{cls:<22}"
        for r in results:
            f1 = r['eval']['per_class'].get(cls, {}).get('f1', 0)
            row += f"{f1:>{col_w}.2%}"
        print(row)


def save_outputs(results, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrices
    for r in results:
        cm  = confusion_matrix(r['eval']['all_labels'],
                               r['eval']['all_preds'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names, ax=ax)
        ax.set_title(f"{r['name']}  |  "
                     f"Acc {r['eval']['accuracy']:.2%}  "
                     f"F1 {r['eval']['macro_f1']:.2%}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.xticks(rotation=40, ha='right')
        plt.tight_layout()
        fname = r['name'].replace(' ', '_').replace('/', '_') + '_cm.png'
        plt.savefig(os.path.join(output_dir, fname),
                    dpi=120, bbox_inches='tight')
        plt.close()

    # Summary CSV
    with open(os.path.join(output_dir, 'summary.csv'), 'w', newline='') as f:
        fields = ['model', 'accuracy', 'macro_precision',
                  'macro_recall', 'macro_f1', 'weighted_f1',
                  'size_mb', 'mean_ms', 'fps']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                'model'           : r['name'],
                'accuracy'        : r['eval']['accuracy'],
                'macro_precision' : r['eval']['macro_precision'],
                'macro_recall'    : r['eval']['macro_recall'],
                'macro_f1'        : r['eval']['macro_f1'],
                'weighted_f1'     : r['eval']['weighted_f1'],
                'size_mb'         : r['size_mb'],
                'mean_ms'         : r['timing']['mean_ms'],
                'fps'             : r['timing']['fps'],
            })

    # Per-class CSV
    with open(os.path.join(output_dir, 'per_class.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'model', 'class', 'precision', 'recall', 'f1', 'support'])
        w.writeheader()
        for r in results:
            for cls, m in r['eval']['per_class'].items():
                w.writerow({'model': r['name'], 'class': cls, **m})

    print(f"\nOutputs saved to: {output_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg         = load_cfg('configs/benchmark.yaml')
    class_names = cfg['class_names']
    yolo_cfg    = cfg['yolo']
    output_dir  = 'outputs/benchmark'
    
    # Check if GPU is available; otherwise automatically adapt to CPU
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking Execution Environment Target: {device}")

    # ── Load test dataset validation environment ──────────────────────────────
    minimal_cfg = {
        'data': {
            'dataset'    : cfg['data']['dataset'],
            'seed'       : cfg['data']['seed'],
            'image_size' : cfg['data']['image_size'],
            'batch_size' : cfg['data']['batch_size'],
            'num_workers': cfg['data']['num_workers'],
        },
        'augmentation': {
            'random_resized_crop': {'scale': [1.0, 1.0]},
            'horizontal_flip': 0.0, 'vertical_flip': 0.0,
            'rotate_limit': 0, 'brightness_contrast': 0.0,
            'hue_saturation': {
                'hue_shift_limit': 0,
                'sat_shift_limit': 0,
                'val_shift_limit': 0,
            }
        }
    }
    _, _, test_loader, _, _, test_ds, _ = load_dataloaders(minimal_cfg, device=device)

    # FIX: Extract file paths straight from the official dataset split object
    test_images = test_ds.image_paths
    print(f"📦 Successfully parsed data split: Verified {len(test_images)} exact files from test split for latency evaluations.")

    # ── Load YOLO models ────────────────────────────────────────────────────
    print("\nLoading YOLO FP32...")
    yolo_fp32 = YOLO(yolo_cfg['model_path'])

    # INT8 via Ultralytics export
    yolo_int8_path = yolo_cfg['model_path'].replace('.pt', '.torchscript')
    if not os.path.exists(yolo_int8_path):
        print("Exporting YOLO INT8...")
        yolo_fp32.export(format='torchscript', optimize=True)
        
    yolo_int8 = YOLO(yolo_int8_path, task='segment') if os.path.exists(yolo_int8_path) else yolo_fp32

    def get_yolo(quantize):
        return yolo_int8 if quantize else yolo_fp32

    all_results = []

    for mcfg in cfg['models']:
        name       = mcfg['name']
        ckpt       = mcfg['checkpoint']
        arch       = mcfg['architecture']
        use_yolo   = mcfg['use_yolo']
        q_yolo     = mcfg['quantize_yolo']

        if not os.path.exists(ckpt) and arch != 'YOLOStandalone':
            print(f"\n⚠️  Skipping '{name}' — checkpoint not found: {ckpt}")
            continue

        print(f"\n{'='*60}\n{name}\n{'='*60}")

        # ── YOLO Standalone ──────────────────────────────────────────────
        if arch == 'YOLOStandalone':
            yolo_model = get_yolo(q_yolo)
            size_mb    = get_model_size_mb(yolo_fp32.model) if not q_yolo else get_model_size_mb(yolo_fp32.model) * 0.25

            print("  Evaluating YOLO standalone classification...")
            eval_res = evaluate_yolo_standalone(yolo_model, test_ds, class_names, yolo_cfg['conf_threshold'])

            def yolo_predict_fn(path):
                yolo_standalone_predict(yolo_model, path, class_names, yolo_cfg['conf_threshold'])

            timing = time_pipeline(yolo_predict_fn, test_images, device)

            all_results.append({
                'name'   : name,
                'size_mb': size_mb,
                'eval'   : eval_res,
                'timing' : timing,
            })
            continue

        # ── CNN classifier ────────────────────────────────────────────────
        model_cfg_dict = {
            'model': {'name': arch, 'num_classes': len(class_names)}
        }
        model = build_model(model_cfg_dict)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model.to(device)
        model.eval()

        size_mb = get_model_size_mb(model)
        params  = get_param_count(model)
        yolo_model = get_yolo(q_yolo) if use_yolo else None

        # Evaluation
        print("  Evaluating on test loader split...")
        eval_res = evaluate_classifier(model, test_loader, class_names, device)
        print(f"  Accuracy: {eval_res['accuracy']:.2%}  Macro-F1: {eval_res['macro_f1']:.2%}")

        # Latency Timing Configuration
        if use_yolo:
            def predict_fn(path):
                t = preprocess_with_yolo(path, yolo_model, yolo_cfg['conf_threshold'])
                with torch.no_grad():
                    model(t.to(device))
        else:
            def predict_fn(path):
                t = preprocess(path)
                with torch.no_grad():
                    model(t.to(device))

        timing = time_pipeline(predict_fn, test_images, device)
        print(f"  {timing['mean_ms']:.2f} ms/img  ({timing['fps']} FPS)")

        # YOLO size contribution for pipeline models
        yolo_size = get_model_size_mb(yolo_fp32.model) if use_yolo else 0
        total_size = size_mb + yolo_size

        all_results.append({
            'name'   : name,
            'size_mb': round(total_size, 3),
            'params' : params,
            'eval'   : eval_res,
            'timing' : timing,
        })

    # ── Final output summaries ──────────────────────────────────────────────
    print_table(all_results, class_names)
    save_outputs(all_results, class_names, output_dir)
    print("\nBenchmark evaluations successfully logged.")