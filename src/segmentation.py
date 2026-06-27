import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image


def load_yolo_segmenter(model_path: str):
    """Load your trained YOLOv8 segmentation model."""
    return YOLO(model_path)


def segment_leaf_yolo(yolo_model, image_np: np.ndarray,
                      fill: str = "white",
                      conf_threshold: float = 0.25) -> np.ndarray:
    """
    Run YOLOv8 segmentation on a single RGB numpy image.
    Returns the image with background removed, filled with
    either white or black.

    Args:
        yolo_model:      loaded YOLO model instance
        image_np:        H x W x 3 uint8 numpy array (RGB)
        fill:            'white' or 'black' background fill
        conf_threshold:  minimum confidence to accept a detection

    Returns:
        H x W x 3 uint8 numpy array with background replaced
    """
    fill_value = (255, 255, 255) if fill == "white" else (0, 0, 0)

    # YOLO expects BGR for cv2-style input, but ultralytics handles
    # numpy RGB arrays directly
    results = yolo_model(image_np, conf=conf_threshold, verbose=False)

    result  = results[0]

    # No detection — return fill-color image rather than crashing
    if result.masks is None or len(result.masks) == 0:
        fallback = np.full_like(image_np, fill_value)
        return fallback

    # If multiple detections, take the one with the highest confidence
    # (most likely the primary leaf in frame)
    best_idx  = int(result.boxes.conf.argmax())
    mask_data = result.masks.data[best_idx].cpu().numpy()

    # Resize mask to match original image dimensions (YOLO resizes internally)
    h, w = image_np.shape[:2]
    mask = cv2.resize(mask_data, (w, h),
                      interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8)  # binary: 1=leaf, 0=background

    # Apply mask: keep leaf pixels, fill background
    output = image_np.copy()
    bg_mask = mask == 0
    output[bg_mask] = fill_value

    return output