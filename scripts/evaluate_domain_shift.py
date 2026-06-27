# scripts/evaluate_domain_shift.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.utils   import load_config, set_seed, get_device
from src.model   import build_model
from src.dataset import get_transforms, RubberLeafDataset
from torch.utils.data import DataLoader

from rembg import remove
from PIL import Image

def segment_leaf_rembg(pil_image):
    output_rgba = remove(pil_image)
    white_bg    = Image.new("RGB", output_rgba.size, (255, 255, 255))
    white_bg.paste(output_rgba, mask=output_rgba.split()[3])
    return white_bg

# ── Class mapping between datasets ────────────────────────────────────────────
PH_TO_BD_MAP = {
    "NO DISEASE"    : {"bd_index": 2, "bd_name": "Healthy"},
    "LEAF SPOT"     : {"bd_index": 3, "bd_name": "Leaf_Spot"},
    "ALGAL SPOT"    : {"bd_index": None, "bd_name": None},
    "POWDERY MILDEW": {"bd_index": None, "bd_name": None},
}

BD_CLASS_NAMES = ["Anthracnose", "Dry_Leaf", "Healthy", "Leaf_Spot"]

# ── Option B: Raw Inference ───────────────────────────────────────────────────
def run_option_b_raw_inference(model, ph_data, ph_names, eval_tf, device):
    print("=" * 60)
    print("OPTION B: Full inference on all PH images (RAW)")
    print("(shows what model predicts for unknown classes)")
    print("=" * 60)

    all_ph_true_names = []
    all_bd_pred_names = []
    all_confidences   = []

    with torch.no_grad():
        for sample in tqdm(ph_data, desc="Option B — Raw inference"):
            img_np  = np.array(sample["image"].convert("RGB"))
            
            # Local tensor creation
            tensor  = eval_tf(image=img_np)["image"].unsqueeze(0).to(device)
            probs   = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

            ph_name  = ph_names[sample["label"]]
            all_ph_true_names.append(ph_name)
            all_bd_pred_names.append(BD_CLASS_NAMES[pred_idx])
            all_confidences.append(float(probs[pred_idx]))

    print("\nPrediction distribution per PH class:")
    print("-" * 60)
    for ph_class in ph_names:
        indices  = [i for i, n in enumerate(all_ph_true_names) if n == ph_class]
        preds    = [all_bd_pred_names[i] for i in indices]
        counts   = Counter(preds)
        total    = len(indices)
        avg_conf = np.mean([all_confidences[i] for i in indices])
        
        print(f"\n  PH class: '{ph_class}' ({total} images, avg confidence: {avg_conf:.2f})")
        for bd_class, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    → Predicted '{bd_class}': {count}/{total} ({100*count/total:.1f}%)")

# ── Option A: Segmented Inference ─────────────────────────────────────────────
def run_option_a_segmented_inference(model, ph_data, ph_names, eval_tf, device, output_dir):
    print("\n" + "=" * 60)
    print("OPTION A: Overlapping classes only (SEGMENTED)")
    print("(Healthy ↔ NO DISEASE  |  Leaf_Spot ↔ LEAF SPOT)")
    print("=" * 60)

    overlap_true = []
    overlap_pred = []

    for sample in tqdm(ph_data, desc="Option A — rembg segmentation"):
        ph_name    = ph_names[sample["label"]]
        mapping    = PH_TO_BD_MAP.get(ph_name)

        if mapping is None or mapping["bd_index"] is None:
            continue

        # Load, segment, and convert
        img_pil  = sample["image"].convert("RGB")
        img_pil  = segment_leaf_rembg(img_pil)           
        img_np   = np.array(img_pil)   

        # [FIXED]: Rebuild tensor locally from the newly segmented numpy array!
        tensor = eval_tf(image=img_np)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            probs    = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            
        pred_idx = int(np.argmax(probs))
        overlap_true.append(mapping["bd_index"])
        overlap_pred.append(pred_idx)

    overlap_class_names = ["Healthy", "Leaf_Spot"]
    remap = {2: 0, 3: 1}
    
    true_remapped = [remap[t] for t in overlap_true]
    pred_remapped = [remap.get(p, -1) for p in overlap_pred]

    total_overlap = len(true_remapped)
    out_of_scope  = sum(1 for p in pred_remapped if p == -1)

    print(f"\nOverlapping test images: {total_overlap}")
    print(f"Predicted outside scope (Anthracnose/Dry_Leaf): "
          f"{out_of_scope} ({100*out_of_scope/total_overlap:.1f}%)\n")

    in_scope_true = [t for t, p in zip(true_remapped, pred_remapped) if p != -1]
    in_scope_pred = [p for p in pred_remapped if p != -1]

    if in_scope_true:
        print("Classification Report (in-scope predictions only):")
        print(classification_report(in_scope_true, in_scope_pred,
                                    target_names=overlap_class_names))

    extended_names = ["Healthy", "Leaf_Spot", "Anthracnose\n(wrong)", "Dry_Leaf\n(wrong)"]
    remap_extended = {2: 0, 3: 1, 0: 2, 1: 3}
    pred_extended  = [remap_extended.get(p, p) for p in overlap_pred]

    cm = confusion_matrix(true_remapped,
                          [remap_extended.get(p) for p in overlap_pred],
                          labels=[0, 1, 2, 3])

    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=extended_names,
                yticklabels=overlap_class_names)
    plt.title("Domain Shift Test — PH-Labeled vs BD-Trained Model\n"
              "Rows: True PH class | Cols: BD model prediction")
    plt.xlabel("Predicted (BD model)")
    plt.ylabel("True (PH class)")
    plt.tight_layout()
    
    path = os.path.join(output_dir, "domain_shift_confusion.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")

# ── Main Execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config("configs/baseline_cnn.yaml")
    set_seed(config["data"]["seed"])
    device = get_device()
    output_dir = os.path.join(config["paths"]["outputs"], "domain_shift")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load BD-trained model
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(config["paths"]["checkpoint"], map_location=device))
    model.eval()
    print(f"Loaded: {config['paths']['checkpoint']}\n")

    # 2. Load PH-Labeled dataset
    print("Loading PH-Labeled dataset...")
    ds_ph    = load_dataset("dffesalbon/rubber-tree-leaf-disease-ph-labeled", token=True)
    ph_data  = ds_ph["train"]
    ph_names = ph_data.features["label"].names
    print(f"PH classes: {ph_names}")
    print(f"Total PH images: {len(ph_data)}\n")

    eval_tf  = get_transforms(config, "eval")

    # 3. Execute isolated inference runs
    run_option_b_raw_inference(model, ph_data, ph_names, eval_tf, device)
    run_option_a_segmented_inference(model, ph_data, ph_names, eval_tf, device, output_dir)