# scripts/prepare_weights.py
import os
from ultralytics import YOLO
import onnx
from onnxconverter_common import float16

def pre_compile_edge_yolo():
    model_path = "checkpoints/yolo_best.pt"
    fp32_onnx = "checkpoints/yolo_best.onnx"
    fp16_onnx = "checkpoints/yolo_best_fp16.onnx"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Source weights not found at {model_path}")
        return
        
    print("🔥 Step 1: Exporting clean ONNX graph via Ultralytics...")
    model = YOLO(model_path)
    
    # Export standard baseline ONNX
    model.export(format='onnx', simplify=True, device=0)
    
    print("\n🗜️ Step 2: Forcing True FP16 Binary Quantization...")
    # Load the exported 22MB FP32 model
    onnx_model = onnx.load(fp32_onnx)
    
    # Cast all internal tensor weights to 16-bit floating point
    onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
    
    # Save the strictly compressed model
    onnx.save(onnx_model_fp16, fp16_onnx)
    
    if os.path.exists(fp16_onnx):
        file_size_mb = os.path.getsize(fp16_onnx) / (1024 * 1024)
        print(f"✅ Success: Static edge weights compiled cleanly -> {fp16_onnx}")
        print(f"📦 Final Edge File Size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    pre_compile_edge_yolo()