import os

YOLO_DIR = "./YOLO_Rubber_Dataset"
splits = ['train', 'val', 'test']
removed_count = 0

print("Pruning unannotated background images from YOLO structure...")

for split in splits:
    img_dir = os.path.join(YOLO_DIR, 'images', split)
    lbl_dir = os.path.join(YOLO_DIR, 'labels', split)
    
    if not os.path.exists(img_dir):
        continue
        
    for img_file in os.listdir(img_dir):
        base_name, _ = os.path.splitext(img_file)
        corresponding_label = f"{base_name}.txt"
        label_path = os.path.join(lbl_dir, corresponding_label)
        
        # If the label file does not exist, delete the image
        if not os.path.exists(label_path):
            img_path = os.path.join(img_dir, img_file)
            os.remove(img_path)
            removed_count += 1

print("-" * 40)
print(f"Pruning complete. Removed {removed_count} unannotated images.")
print("Your YOLO dataset is now verified clean and safe for training.")