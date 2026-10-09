import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def export_model():
    print("Model Export Utility Initialized.")
    print("---------------------------------")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[WARNING] 'ultralytics' is not installed.")
        print("To export a PyTorch YOLOv8 model, run:")
        print("    pip install ultralytics")
        print("\nShowing export code template below:\n")
        
        # Example code template
        code = """
        # Python code to export YOLOv8 to ONNX and TensorFlow.js:
        from ultralytics import YOLO
        
        # 1. Load trained PyTorch model
        model = YOLO("runs/detect/train/weights/best.pt")
        
        # 2. Export to ONNX format (for server-side Fastapi/ONNX Runtime)
        onnx_path = model.export(format="onnx")
        print("ONNX model saved at:", onnx_path)
        
        # 3. Export to TensorFlow.js format (for browser-side inference)
        # Note: requires pip install tensorflowjs
        tfjs_path = model.export(format="tfjs")
        print("TensorFlow.js model saved at:", tfjs_path)
        """
        print(code)
        return

    # If ultralytics is installed, attempt to export a dummy or trained model
    weights_path = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not os.path.exists(weights_path):
        print(f"Error: Trained weights not found at '{weights_path}'. Please train the model first.")
        return
        
    model = YOLO(weights_path)
    
    print(f"Exporting '{weights_path}' to ONNX...")
    model.export(format="onnx")
    
    print(f"Exporting '{weights_path}' to TFJS...")
    model.export(format="tfjs")
    print("Export complete.")

if __name__ == "__main__":
    export_model()
