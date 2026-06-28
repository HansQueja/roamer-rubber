# main.py
import os
import argparse
from src.pipeline import RoamerPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="ROAMER Autonomous Disease Inference Loop")
    parser.add_argument("--image", type=str, required=True, 
                        help="Path to the input leaf image or directory of images")
    parser.add_argument("--config", type=str, default="configs/baseline_cnn.yaml", 
                        help="Path to the model configuration YAML file")
    parser.add_argument("--yolo", type=str, default="checkpoints/yolo_best.pt", 
                        help="Path to the trained YOLOv8 segmenter weights")
    parser.add_argument("--classifier", type=str, default="checkpoints/best_deepenhanced_cnn.pth", 
                        help="Path to the DeepEnhancedCNN_SE weights")
    parser.add_argument("--cam_threshold", type=float, default=0.35, 
                        help="Grad-CAM activation threshold for spatial ROI gating")
    return parser.parse_args()

def process_single_image(pipeline, image_path, cam_threshold):
    print(f"\nProcessing: {os.path.basename(image_path)}")
    print("-" * 50)
    
    try:
        # Run the full 3-stage inference pipeline
        results = pipeline.run_inference(image_path, cam_threshold=cam_threshold)
        
        if not results['leaf_detected']:
            print("❌ Target Warning: No leaf structure detected by YOLO segmenter.")
            print(f"Fallback Diagnosis: {results['diagnosis']} (Confidence: {results['confidence']:.2%})")
            return

        # 1. Print Core Classification Metrics
        print(f"📋 Diagnosis  :  {results['diagnosis']}")
        print(f"🎯 Confidence :  {results['confidence']:.2%}")
        
        # 2. Print Comparative Severity Metrics Safely
        # Handles the 'None' primitive conditions for Healthy and Dry_Leaf classes
        def format_severity(val):
            return f"{val}%" if val is not None else "None (Excluded)"

        print(f"📸 Severity (Grad-CAM Coarse) : {format_severity(results['severity_gradcam'])}")
        print(f"🎨 Severity (Pure HSV Color)  : {format_severity(results['severity_color'])}")
        print(f"🧬 Severity (Hybrid Guided)   : {format_severity(results['severity_final'])}")
        
        # 3. Optional: Print top-3 class distribution breakdown for edge debugging
        print("\nProbability Distribution Breakdown:")
        sorted_probs = sorted(results['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for cls_name, prob in sorted_probs[:3]:
            print(f"  └─ {cls_name}: {prob:.2%}")
            
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
    except Exception as e:
        print(f"❌ Execution Crash during inference pipeline: {e}")

def main():
    args = parse_args()

    # Initialize the Two-Stage + Quantification Pipeline once (keeps models in memory)
    print("🤖 Initializing ROAMER Inspection Cascade...")
    pipeline = RoamerPipeline(
        config_path=args.config,
        yolo_path=args.yolo,
        classifier_path=args.classifier
    )
    print("✓ All models mapped to hardware and ready.")

    # Determine if input path is a single file or a folder directory
    if os.path.isfile(args.image):
        process_single_image(pipeline, args.image, args.cam_threshold)
    elif os.path.isdir(args.image):
        print(f"\n📂 Scanning directory batch: {args.image}")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.BMP', '.tiff')
        images = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.endswith(valid_extensions)]
        
        if not images:
            print("No valid image formats found in directory.")
            return
            
        print(f"Found {len(images)} targets. Commencing batch extraction...")
        for img_path in images:
            process_single_image(pipeline, img_path, args.cam_threshold)
    else:
        print(f"❌ Input target path error: '{args.image}' is neither a valid file nor directory.")

if __name__ == "__main__":
    main()