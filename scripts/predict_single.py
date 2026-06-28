import os
import cv2
import time
import torch
import numpy as np
import argparse
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your final 7-layer SE architecture
from src.model import DeepEnhancedCNN_SE

def parse_args():
    parser = argparse.ArgumentParser(description="Test YOLO -> SE-CNN Classification Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    parser.add_argument("--yolo", type=str, default="checkpoints/yolo_best.pt", help="YOLO segmenter weights")
    parser.add_argument("--cnn", type=str, default="checkpoints/best_deepenhanced_cnn.pth", help="SE-CNN classifier weights")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Hardware Mapping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚙️  Initializing on {device}...")

    # 2. Load Models
    print("📦 Loading YOLOv8s Segmenter...")
    yolo_model = YOLO(args.yolo)

    print("📦 Loading DeepEnhancedCNN_SE Classifier...")
    num_classes = 6
    class_names = ['Algal_Spot', 'Anthracnose', 'Dry_Leaf', 'Healthy', 'Leaf_Spot', 'Powdery_Mildew']
    
    cnn_model = DeepEnhancedCNN_SE(num_classes).to(device)
    cnn_model.load_state_dict(torch.load(args.cnn, map_location=device))
    cnn_model.eval()

    # 3. Preprocessing (Must match training: 224x224, standard ImageNet normalization)
    preprocess = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 4. Load Image
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found at {args.image}")
        return
        
    raw_bgr = cv2.imread(args.image)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    h, w = raw_rgb.shape[:2]

    print(f"\n🔍 Processing: {os.path.basename(args.image)}")
    print("-" * 40)
    
    start_time = time.time()

    # --- STAGE 1: YOLO Segmentation ---
    yolo_start = time.time()
    results = yolo_model(raw_rgb, conf=0.10, verbose=False)[0]
    
    if results.masks is None or len(results.masks) == 0:
        print("⚠️  No leaf detected by YOLO. Exiting pipeline.")
        return

    # Extract the highest confidence mask
    best_idx = int(results.boxes.conf.argmax())
    mask_data = results.masks.data[best_idx].cpu().numpy()
    mask = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
    leaf_mask = (mask > 0.5).astype(np.uint8) * 255

    # Apply the Black Fill (The 95% Accuracy Fix)
    isolated_leaf = raw_rgb.copy()
    isolated_leaf[leaf_mask == 0] = (0, 0, 0)
    yolo_time = time.time() - yolo_start

    # --- STAGE 2: SE-CNN Classification ---
    cnn_start = time.time()
    
    # Preprocess the isolated, black-filled image
    input_tensor = preprocess(image=isolated_leaf)['image'].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = cnn_model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    cnn_time = time.time() - cnn_start
    
    total_time = time.time() - start_time

    # 5. Output Results
    print(f"✅ Prediction : {pred_name}")
    print(f"🎯 Confidence : {confidence:.2%}")
    print("-" * 40)
    print("⏱️  Latency Metrics:")
    print(f"   YOLO Segmentation : {yolo_time * 1000:.1f} ms")
    print(f"   CNN Forward Pass  : {cnn_time * 1000:.1f} ms")
    print(f"   Total Pipeline    : {total_time * 1000:.1f} ms")
    print("-" * 40)
    
    # Optional: Save the intermediate black-filled image to verify what the CNN saw
    debug_path = "debug_isolated_input.jpg"
    cv2.imwrite(debug_path, cv2.cvtColor(isolated_leaf, cv2.COLOR_RGB2BGR))
    print(f"📸 Saved CNN input payload to: {debug_path}")

if __name__ == "__main__":
    main()