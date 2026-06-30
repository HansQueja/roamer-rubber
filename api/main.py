# api/main.py
import os
import tempfile
import base64
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Import YOUR existing pipeline
from src.pipeline import RoamerPipeline

app = FastAPI(title="ROAMER Disease Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize your Two-Stage Pipeline Globally (Runs once on server startup)
print("🤖 Initializing ROAMER Inspection Cascade...")
pipeline = RoamerPipeline(
    config_path="configs/baseline_cnn.yaml",  # Ensure this points to the right config
    yolo_path="checkpoints/yolo_best.pt",
    classifier_path="checkpoints/best_deepenhanced_cnn.pth"
)
print("✓ All models mapped to hardware and ready.")

def encode_image_to_base64(image_array):
    """Converts numpy image arrays from your pipeline to base64 for the Next.js frontend."""
    if image_array is None:
        return None
    # Convert RGB (from pipeline) to BGR for cv2 encoding
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/analyze")
async def analyze_leaf(file: UploadFile = File(...)):
    # 2. Save the uploaded web image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 3. Run your exact inference loop!
        results = pipeline.run_inference(tmp_path, cam_threshold=0.35)
        
        # Clean up the temporary file immediately
        os.remove(tmp_path)

        if not results.get('leaf_detected', True):
            return {"error": "No leaf structure detected by YOLO segmenter."}

        # 4. Safely extract your severity metrics (handles 'None' for Healthy/Dry_Leaf)
        sev_gradcam = results.get('severity_gradcam')
        sev_color = results.get('severity_color')
        sev_final = results.get('severity_final')

        # 5. Extract Images 
        # *Note: Ensure your pipeline.run_inference() is returning the numpy 
        # arrays for the segmented and XAI images in its results dictionary!*
        segmented_b64 = encode_image_to_base64(results.get('segmented_image'))
        xai_b64 = encode_image_to_base64(results.get('xai_image'))

        # 6. Package it for the Next.js Frontend
        return {
            "diagnosis": results.get('diagnosis', 'Unknown').replace("_", " "),
            "confidence": round(results.get('confidence', 0) * 100, 2),
            "severity_gradcam": sev_gradcam,
            "severity_color": sev_color,
            "severity_final": sev_final,
            "segmented_image": segmented_b64,
            "xai_image": xai_b64
        }

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"error": f"Execution Crash during inference pipeline: {str(e)}"}