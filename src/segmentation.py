import numpy as np
import cv2
from ultralytics import YOLO
import os


def load_yolo_segmenter(model_path: str):
    """Load your trained YOLOv8 segmentation model."""
    return YOLO(model_path)


def segment_leaf_yolo(yolo_model, image_np, fill="white",
                      conf_threshold=0.25):
    fill_value = (255, 255, 255) if fill == "white" else (0, 0, 0)
    h, w       = image_np.shape[:2]

    results = yolo_model(image_np, conf=conf_threshold, verbose=False)
    result  = results[0]

    if result.masks is None or len(result.masks) == 0:
        # Return original image unchanged rather than a white blank
        # A failed segmentation is less harmful than a meaningless input
        return image_np

    best_idx  = int(result.boxes.conf.argmax())
    mask_data = result.masks.data[best_idx].cpu().numpy()
    mask      = cv2.resize(mask_data, (w, h),
                           interpolation=cv2.INTER_NEAREST)
    mask      = (mask > 0.5).astype(np.uint8)

    output           = image_np.copy()
    output[mask == 0] = fill_value
    return output


def presegment_dataset(yolo_model, image_paths, output_dir,
                       fill="white", conf_threshold=0.25):
    """
    Run YOLO segmentation over all images once and save results to disk.
    Training then loads from output_dir instead of running YOLO live.

    Args:
        yolo_model:     loaded YOLO instance
        image_paths:    list of source image paths (from split.csv)
        output_dir:     where to save segmented images
        fill:           'white' or 'black'
        conf_threshold: YOLO confidence threshold

    Returns:
        List of output paths in the same order as image_paths
    """
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    output_paths  = []
    no_detect     = 0

    for src_path in tqdm(image_paths, desc="Pre-segmenting dataset"):
        # Preserve subfolder structure: Leaf_Spot/img.jpg → output_dir/Leaf_Spot/img.jpg
        rel_path   = os.path.relpath(src_path,
                                     start=os.path.commonpath(image_paths))
        dst_path   = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if os.path.exists(dst_path):
            # Skip if already segmented (allows resuming interrupted runs)
            output_paths.append(dst_path)
            continue

        image = cv2.imread(src_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmented = segment_leaf_yolo(yolo_model, image,
                                      fill=fill,
                                      conf_threshold=conf_threshold)

        if np.array_equal(segmented, image):
            no_detect += 1   # original returned — YOLO found nothing

        # Save as RGB (cv2 expects BGR)
        cv2.imwrite(dst_path,
                    cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
        output_paths.append(dst_path)

    print(f"Pre-segmentation complete. "
          f"No-detection fallbacks: {no_detect}/{len(image_paths)} "
          f"({100*no_detect/len(image_paths):.1f}%)")
    return output_paths