import os
import cv2
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

# ─── 1. Configuration ───
CSV_PATH = "../Unified_Dataset/split.csv"
SOURCE_DIR = "../Unified_Dataset"
OUTPUT_DIR = "./YOLO_Rubber_Dataset"
MANUAL_DIR = "./Needs_Manual_Annotation"

# ─── 2. Utility Functions ───
def target_bound(val):
    """Ensures coordinates strictly stay within 0.0 and 1.0"""
    return max(0.0, min(1.0, val))

def convert_contour_to_yolo(contour, img_width, img_height):
    """Normalizes OpenCV contour pixels into YOLOv8-Seg format."""
    contour_squeezed = contour.squeeze()
    if contour_squeezed.ndim != 2 or len(contour_squeezed) < 3:
        return None
        
    normalized_coords = []
    for pt in contour_squeezed:
        norm_x = pt[0] / img_width
        norm_y = pt[1] / img_height
        normalized_coords.append(f"{target_bound(norm_x):.6f} {target_bound(norm_y):.6f}")
        
    return " ".join(normalized_coords)

# ─── 3. Build Directory Structures ───
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
    os.makedirs(os.path.join(MANUAL_DIR, split), exist_ok=True)

# ─── 4. Load Data and Process ───
df = pd.read_csv(CSV_PATH)
print(f"Building YOLO dataset and extracting orphans for {len(df)} images...")

success_count = 0
fail_count = 0
moved_count = {'train': 0, 'val': 0, 'test': 0}

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    split = row['split']
    class_id = row['label']
    src_rel_path = row['file_path']
    src_full_path = os.path.join(SOURCE_DIR, src_rel_path)
    
    filename = os.path.basename(src_full_path)
    filename_no_ext, ext = os.path.splitext(filename)
    
    dest_img_path = os.path.join(OUTPUT_DIR, 'images', split, filename)
    dest_label_path = os.path.join(OUTPUT_DIR, 'labels', split, f"{filename_no_ext}.txt")
    
    # Copy the image to the YOLO folder initially
    if not os.path.exists(dest_img_path):
        shutil.copy(src_full_path, dest_img_path)
        
    img_np = cv2.imread(src_full_path)
    if img_np is None:
        fail_count += 1
        continue
        
    img_height, img_width = img_np.shape[:2]
    contour = None
    
    # --- The Robust Extraction Logic ---
    if filename_no_ext.endswith("_SE"):
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            
    elif filename_no_ext.endswith("_BD"):
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # 1. Use Otsu's thresholding for smart background separation
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. THE BORDER WIPE: Force the outer 10 pixels to black
        mask[0:10, :] = 0          # Top edge
        mask[-10:, :] = 0          # Bottom edge
        mask[:, 0:10] = 0          # Left edge
        mask[:, -10:] = 0          # Right edge
        
        # 3. Clean up internal noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            
    elif filename_no_ext.endswith("_LB"):
        twin_path = src_full_path.replace("_LB", "_SE")
        
        # SAFETY CHECK: Only attempt to read if the twin actually exists
        if os.path.exists(twin_path):
            twin_np = cv2.imread(twin_path)
            if twin_np is not None:
                gray = cv2.cvtColor(twin_np, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
    
    # --- Save the Label OR Route to Manual Directory ---
    label_saved = False
    if contour is not None:
        yolo_coords = convert_contour_to_yolo(contour, img_width, img_height)
        if yolo_coords:
            with open(dest_label_path, "w") as f:
                f.write(f"{class_id} {yolo_coords}\n")
            success_count += 1
            label_saved = True

    # If the process failed to generate a label file
    if not label_saved:
        fail_count += 1
        orphan_dest_path = os.path.join(MANUAL_DIR, split, filename)
        
        # Move the image OUT of the YOLO directory and into the Manual directory
        if os.path.exists(dest_img_path):
            shutil.move(dest_img_path, orphan_dest_path)
            moved_count[split] += 1

print("\n" + "="*40)
print("PROCESSING COMPLETE")
print("="*40)
print(f"Successfully generated polygons for: {success_count} images")
print(f"Failed extractions isolated:         {fail_count} images\n")

print("Breakdown of images requiring manual annotation:")
for s, count in moved_count.items():
    if count > 0:
        print(f" - {s.capitalize()}: {count} images")

print(f"\nYour clean training data is in: {OUTPUT_DIR}")
print(f"Your manual annotation batches are in: {MANUAL_DIR}")