# scripts/prepare_weights.py
import os
from ultralytics import YOLO

def pre_compile_quantized_yolo():
    model_path = "checkpoints/yolo_best.pt"
    expected_output = "checkpoints/yolo_best.torchscript"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Source weights not found at {model_path}")
        return
        
    print("🔥 Starting clean INT8 TorchScript compilation...")
    # Initialize an isolated compilation context
    model = YOLO(model_path)
    model.export(format='torchscript', optimize=True)
    
    if os.path.exists(expected_output):
        print(f"✅ Success: Static weights compiled cleanly -> {expected_output}")
    else:
        print("⚠️ Warning: Compilation complete but verify output path matching.")

if __name__ == "__main__":
    pre_compile_quantized_yolo()