# scripts/benchmark.py
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

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

_eval_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return _eval_tf(image=img)['image'].unsqueeze(0)

def segment_with_yolo(yolo_model, image_rgb, conf=0.10):
    h, w = image_rgb.shape[:2]
    result = yolo_model(image_rgb, conf=conf, verbose=False)[0]

    if result.masks is None or len(result.masks) == 0:
        return image_rgb   # Fallback

    best = int(result.boxes.conf.argmax())
    mask = cv2.resize(result.masks.data[best].cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8)
    out = image_rgb.copy()
    out[mask == 0] = 0  # Clean black fill
    return out

def preprocess_with_yolo(path, yolo_model, conf=0.10):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    segmented = segment_with_yolo(yolo_model, img, conf)
    return _eval_tf(image=segmented)['image'].unsqueeze(0)

def yolo_standalone_predict(yolo_model, path, class_names, conf=0.10):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    result = yolo_model(img, conf=conf, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        return 3, 1.0  # Default Healthy prediction fallback

    best = int(result.boxes.conf.argmax())
    cls_idx = int(result.boxes.cls[best].item())
    return cls_idx, float(result.boxes.conf[best].item())

# ── Ground Truth Evaluation Loops ───────────────────────────────────────────

def evaluate_classifier_pipeline(model, test_images, test_labels, class_names, device, yolo_model=None, conf=0.10):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for path in test_images:
            if yolo_model is not None:
                t = preprocess_with_yolo(path, yolo_model, conf)
            else:
                t = preprocess(path)
            
            preds = torch.argmax(model(t.to(device)), dim=1).cpu().numpy()[0]
            all_preds.append(preds)
            
    return _compute_metrics(np.array(all_preds), np.array(test_labels), class_names)

def evaluate_yolo_standalone(yolo_model, test_images, test_labels, class_names, conf=0.10):
    all_preds = []
    for path in test_images:
        pred_idx, _ = yolo_standalone_predict(yolo_model, path, class_names, conf)
        all_preds.append(pred_idx)
    return _compute_metrics(np.array(all_preds), np.array(test_labels), class_names)

def _compute_metrics(preds, labels, class_names):
    acc = accuracy_score(labels, preds)
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(class_names))), zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)

    per_class = {
        class_names[i]: {
            'precision': round(float(p[i]), 4),
            'recall': round(float(r[i]), 4),
            'f1': round(float(f1[i]), 4),
            'support': int(support[i]),
        } for i in range(len(class_names))
    }
    return {
        'accuracy': round(float(acc), 4),
        'macro_precision': round(float(mp), 4),
        'macro_recall': round(float(mr), 4),
        'macro_f1': round(float(mf1), 4),
        'weighted_f1': round(float(wf1), 4),
        'per_class': per_class,
        'all_preds': preds,
        'all_labels': labels,
    }

def time_pipeline(predict_fn, image_paths, device, warmup=5):
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
        'mean_ms': round(float(timings.mean()), 3),
        'fps': round(1000 / float(timings.mean()), 2),
    }

def print_table(results, class_names):
    hdr = f"{'Model Configuration':<38} {'Acc':>7} {'mP':>6} {'mR':>6} {'mF1':>6} {'wF1':>6} {'Size(MB)':>8} {'ms/img':>8} {'FPS':>7}"
    div = "─" * len(hdr)
    print(f"\n{div}\nROAMER END-TO-END BENCHMARK RESULTS\n{div}\n{hdr}\n{div}")
    for r in results:
        ev, tm = r['eval'], r['timing']
        print(f"{r['name']:<38} {ev['accuracy']:>7.2%} {ev['macro_precision']:>6.2%} {ev['macro_recall']:>6.2%} {ev['macro_f1']:>6.2%} {ev['weighted_f1']:>6.2%} {r['size_mb']:>8.2f} {tm['mean_ms']:>8.2f} {tm['fps']:>7.1f}")
    print(div)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = load_cfg('configs/benchmark.yaml')
    class_names = cfg['class_names']
    yolo_cfg = cfg['yolo']
    output_dir = 'outputs/benchmark'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting Processing Device Core: {device}")

    # ── Load test validation data split structures ──────────────────────────
    minimal_cfg = {
        'data': {
            'dataset': cfg['data']['dataset'],
            'seed': cfg['data']['seed'],
            'image_size': cfg['data']['image_size'],
            'batch_size': cfg['data']['batch_size'],
            'num_workers': cfg['data']['num_workers'],
        },
        'augmentation': {
            'random_resized_crop': {'scale': [1.0, 1.0]}, 'horizontal_flip': 0.0, 'vertical_flip': 0.0, 'rotate_limit': 0, 'brightness_contrast': 0.0,
            'hue_saturation': {'hue_shift_limit': 0, 'sat_shift_limit': 0, 'val_shift_limit': 0}
        }
    }
    _, _, _, _, _, test_ds, _ = load_dataloaders(minimal_cfg, device=device)

    test_images = test_ds.image_paths
    test_labels = test_ds.labels
    print(f" Verified {len(test_images)} raw images extracted from evaluation split.")

    print("Statically initializing validation targets...")
    yolo_fp32 = YOLO(yolo_cfg['fp32_path'])
    
    # Load the ONNX model
    if os.path.exists(yolo_cfg['onnx_path']):
        yolo_edge = YOLO(yolo_cfg['onnx_path'], task='segment')
        print("✓ Loaded static pre-compiled ONNX FP16 model.")
    else:
        print("⚠️ Warning: ONNX file not found. Falling back to FP32.")
        yolo_edge = yolo_fp32

    all_results = []

    for mcfg in cfg['models']:
        name = mcfg['name']
        ckpt = mcfg['checkpoint']
        arch = mcfg['architecture']
        use_yolo = mcfg['use_yolo']
        q_yolo = mcfg['quantize_yolo']

        if not os.path.exists(ckpt) and arch != 'YOLOStandalone':
            continue

        print(f"\nEvaluating: {name}...")
        
        # Use the ONNX model if q_yolo is true
        current_yolo = yolo_edge if q_yolo else yolo_fp32

        # --- Case A: YOLO Standalone Head ---
        if arch == 'YOLOStandalone':
            size_mb = get_model_size_mb(yolo_fp32.model) if not q_yolo else get_model_size_mb(yolo_fp32.model) * 0.25
            eval_res = evaluate_yolo_standalone(current_yolo, test_images, test_labels, class_names, yolo_cfg['conf_threshold'])
            
            def yolo_predict_fn(path):
                yolo_standalone_predict(current_yolo, path, class_names, yolo_cfg['conf_threshold'])
                
            timing = time_pipeline(yolo_predict_fn, test_images, device)
            
            all_results.append({'name': name, 'size_mb': size_mb, 'eval': eval_res, 'timing': timing})
            continue

        # --- Case B: Cascaded Classifiers ---
        model_cfg_dict = {'model': {'name': arch, 'num_classes': len(class_names)}}
        model = build_model(model_cfg_dict)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model.to(device).eval()

        size_mb = get_model_size_mb(model)
        
        # Evaluate accuracy and f1 against the raw dataset structure
        eval_res = evaluate_classifier_pipeline(
            model, test_images, test_labels, class_names, device, 
            yolo_model=current_yolo if use_yolo else None, conf=yolo_cfg['conf_threshold']
        )

        # Measure true hardware latency across processing pipelines
        if use_yolo:
            def predict_fn(path):
                t = preprocess_with_yolo(path, current_yolo, yolo_cfg['conf_threshold'])
                with torch.no_grad():
                    model(t.to(device))
        else:
            def predict_fn(path):
                t = preprocess(path)
                with torch.no_grad():
                    model(t.to(device))

        timing = time_pipeline(predict_fn, test_images, device)

        # Accumulate memory size weights
        yolo_contrib = (get_model_size_mb(yolo_fp32.model) if not q_yolo else (get_model_size_mb(yolo_fp32.model) * 0.25)) if use_yolo else 0
        
        all_results.append({
            'name': name, 'size_mb': round(size_mb + yolo_contrib, 3), 'eval': eval_res, 'timing': timing
        })

    print_table(all_results, class_names)