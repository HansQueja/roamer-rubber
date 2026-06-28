import cv2
import torch
import numpy as np
import albumentations as A

from ultralytics import YOLO
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.model import build_model
from src.utils import load_config


class RoamerPipeline:
    """
    Two-stage inference pipeline:
      Stage 1 — YOLOv8s segments the leaf from background
      Stage 2 — Classifier diagnoses the disease
      Stage 3 — Severity estimated via Grad-CAM guided color segmentation
    """

    # Refined HSV ranges focusing specifically on necrotic/chlorotic lesion hues
    DISEASE_HSV_RANGES = {
        'Anthracnose'    : [(10,  40,  20), (30, 255, 180)],   # Deep dark browns/necrotic holes
        'Leaf_Spot'      : [(10,  50,  40), (28, 255, 220)],   # Yellowish-brown halos and spots
        'Algal_Spot'     : [(5,   60,  60), (22, 255, 200)],   # Rust-orange velvety patches
        'Powdery_Mildew' : [(0,   0,  160), (180, 50, 255)],   # Pale white/grey superficial mycelium
        'Healthy'        : None,
        'Dry_Leaf'       : None,                               # Abiotic/Senescent (Excluded)
    }

    # Define classes that should not compute a disease severity score
    NON_DISEASE_CLASSES = {'Healthy', 'Dry_Leaf'}

    def __init__(self, config_path: str, yolo_path: str, classifier_path: str):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Device: {self.device}")

        # Stage 1 — YOLO leaf segmenter
        print("Loading YOLO segmenter...")
        self.yolo = YOLO(yolo_path)

        # Stage 2 — Disease classifier
        name = self.config['model']['name']
        print(f"Loading {name}...")
        self.classifier = build_model(self.config).to(self.device)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.classifier.eval()

        # Preprocessing — matches your exact training configuration
        self.preprocess = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.target_layers = self._get_target_layers()
        self.class_names = self.config['data'].get(
            'class_names',
            ['Algal_Spot', 'Anthracnose', 'Dry_Leaf', 'Healthy', 'Leaf_Spot', 'Powdery_Mildew']
        )

    def _get_target_layers(self):
        name = self.config['model']['name']
        if name == 'BaselineCNN':
            return [self.classifier.conv3]
        elif name in ('EnhancedCNN', 'DeepEnhancedCNN'):
            return [self.classifier.features[-3]]
        elif name == 'DeepEnhancedCNN_SE':
            # Target the last convolutional layer right before the final SE block/GAP
            return [self.classifier.features[-4]]
        elif name == 'LeafNet':
            return [self.classifier.stage3[-2]]
        elif name == 'MobileNetEdge':
            return [self.classifier.model.features[-1]]
        else:
            raise ValueError(f"No target layer defined for {name}")

    # ── Stage 1: YOLO leaf segmentation ──────────────────────────────────────

    def _segment_leaf(self, image_rgb: np.ndarray, conf_threshold: float = 0.10):
        h, w = image_rgb.shape[:2]
        results = self.yolo(image_rgb, conf=conf_threshold, verbose=False)[0]

        if results.masks is None or len(results.masks) == 0:
            return None, image_rgb

        best_idx = int(results.boxes.conf.argmax())
        mask_data = results.masks.data[best_idx].cpu().numpy()
        mask = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
        leaf_mask = (mask > 0.5).astype(np.uint8) * 255

        isolated = image_rgb.copy()
        isolated[leaf_mask == 0] = (0, 0, 0)

        return leaf_mask, isolated

    # ── Stage 2: Classification ───────────────────────────────────────────────

    def _classify(self, isolated_leaf_rgb: np.ndarray):
        tensor = self.preprocess(image=isolated_leaf_rgb)['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_name = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        return pred_idx, pred_name, confidence, probs, tensor

    # ── Stage 3: Refined Severity Estimation ─────────────────────────────────

    def _estimate_severity_guided(self, input_tensor: torch.Tensor,
                                  pred_idx: int,
                                  isolated_rgb: np.ndarray,
                                  leaf_mask: np.ndarray,
                                  disease_name: str,
                                  cam_threshold: float = 0.35) -> tuple:
        """
        Hybrid Severity: Uses Grad-CAM neural attention maps to isolate the general 
        neighborhood of the infection, then runs fine-grained HSV color thresholding
        exclusively inside that region to get accurate pixel counts.
        """
        # 1. Generate coarse neural attention region via Grad-CAM
        targets = [ClassifierOutputTarget(pred_idx)]
        with GradCAM(model=self.classifier, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        h, w = leaf_mask.shape
        cam_resized = cv2.resize(grayscale_cam, (w, h))
        
        # Binary mask of where the network is paying attention
        attention_mask = (cam_resized > cam_threshold).astype(np.uint8) * 255

        # 2. Extract specific disease color pixels globally
        hsv_range = self.DISEASE_HSV_RANGES.get(disease_name)
        if hsv_range is None:
            return 0.0, 0.0, 0.0

        hsv = cv2.cvtColor(isolated_rgb, cv2.COLOR_RGB2HSV)
        lo, hi = np.array(hsv_range[0]), np.array(hsv_range[1])
        global_color_mask = cv2.inRange(hsv, lo, hi)

        # 3. Intersect constraints to eliminate false positives
        leaf_binary = (leaf_mask > 0).astype(np.uint8)
        
        # Isolated methods for benchmarking
        disease_cam_only = (attention_mask > 0).astype(np.uint8) & leaf_binary
        disease_color_only = (global_color_mask > 0).astype(np.uint8) & leaf_binary
        
        # Guided Hybrid Intersection: Must be a disease color AND inside the network's attention zone
        disease_guided = disease_color_only & (attention_mask > 0)

        # 4. Compute fractional areas
        total_leaf_px = int(leaf_binary.sum())
        if total_leaf_px == 0:
            return 0.0, 0.0, 0.0

        sev_cam = round((int(disease_cam_only.sum()) / total_leaf_px) * 100, 2)
        sev_color = round((int(disease_color_only.sum()) / total_leaf_px) * 100, 2)
        sev_guided = round((int(disease_guided.sum()) / total_leaf_px) * 100, 2)

        return sev_cam, sev_color, min(sev_guided, 100.0)

    # ── Full inference ─────────────────────────────────────────────────────────

    def run_inference(self, image_path: str, cam_threshold: float = 0.35) -> dict:
        raw_bgr = cv2.imread(image_path)
        if raw_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        # Stage 1 — Segment Leaf Boundary
        leaf_mask, isolated = self._segment_leaf(raw_rgb)
        leaf_detected = leaf_mask is not None

        if not leaf_detected:
            return {
                'diagnosis': 'Healthy',
                'confidence': 1.0,
                'all_probabilities': {n: 0.0 for n in self.class_names},
                'severity_gradcam': None,
                'severity_color': None,
                'severity_final': None,
                'leaf_detected': False,
            }

        # Stage 2 — Deep Enhanced CNN Classification
        pred_idx, pred_name, confidence, probs, input_tensor = self._classify(isolated)

        # Stage 3 — Severity Enforcement
        if pred_name in self.NON_DISEASE_CLASSES:
            # Enforce 'None' layout explicitly for healthy or senescent leaves
            sev_cam = None
            sev_color = None
            sev_final = None
        else:
            sev_cam, sev_color, sev_final = self._estimate_severity_guided(
                input_tensor, pred_idx, isolated, leaf_mask, pred_name, cam_threshold
            )

        return {
            'diagnosis': pred_name,
            'confidence': round(confidence, 4),
            'all_probabilities': {n: round(float(p), 4) for n, p in zip(self.class_names, probs)},
            'severity_gradcam': sev_cam,
            'severity_color': sev_color,
            'severity_final': sev_final,
            'leaf_detected': True,
        }