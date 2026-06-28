# scripts/prepare_weights.py
import os
from ultralytics import YOLO

def pre_compile_edge_yolo():
    model_path = "checkpoints/yolo_best.pt"
    expected_output = "checkpoints/yolo_best.onnx"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Source weights not found at {model_path}")
        return
        
    print("🔥 Starting clean ONNX FP16 compilation on GPU...")
    model = YOLO(model_path)
    
    # UPDATED API: quantize=16 instead of half=True
    model.export(format='onnx', quantize=16, simplify=True, device=0)
    
    if os.path.exists(expected_output):
        file_size_mb = os.path.getsize(expected_output) / (1024 * 1024)
        print(f"✅ Success: Static weights compiled cleanly -> {expected_output}")
        print(f"📦 Final Edge File Size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    pre_compile_edge_yolo()