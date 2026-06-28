# scripts/prepare_weights.py
import os
from ultralytics import YOLO

def pre_compile_edge_yolo():
    model_path = "checkpoints/yolo_best.pt"
    expected_output = "checkpoints/yolo_best.onnx"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Source weights not found at {model_path}")
        return
        
    print("🔥 Starting clean ONNX FP16 compilation...")
    model = YOLO(model_path)
    
    # half=True cuts the size in half (FP16)
    # simplify=True removes redundant computational nodes
    model.export(format='onnx', half=True, simplify=True, device=0)
    
    if os.path.exists(expected_output):
        print(f"✅ Success: Static weights compiled cleanly -> {expected_output}")

if __name__ == "__main__":
    pre_compile_edge_yolo()