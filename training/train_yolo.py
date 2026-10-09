import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def train_yolo():
    print("YOLOv8 Banknote Training Script Initialized.")
    print("--------------------------------------------")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[WARNING] 'ultralytics' package is not installed. To run YOLO training:")
        print("    pip install ultralytics torch torchvision")
        print("\nContinuing execution to show template code...\n")
        return

    # Load a pre-trained YOLOv8 nano model (lightweight, runs fast on CPUs and edge devices)
    model = YOLO("yolov8n.pt")
    
    # Dataset YAML path
    dataset_yaml = PROJECT_ROOT / "data" / "dataset.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: dataset config not found at '{dataset_yaml}'. Please run download_dataset.py first.")
        sys.exit(1)
        
    print(f"Starting YOLOv8 training on dataset: {dataset_yaml}")
    
    # Train the model
    # epochs=20 is quick, workers=2 avoids memory overheads, imgsz=640 is standard
    results = model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=640,
        batch=16,
        device="cpu",  # Change to "0" or "cuda" if GPU is available
        workers=2
    )
    
    print("Training finished! Results saved in runs/detect/train/")
    
    # Export the model to ONNX format for easy loading in FastAPI or TFJS
    print("Exporting model to ONNX...")
    path = model.export(format="onnx")
    print(f"Model exported to: {path}")

if __name__ == "__main__":
    train_yolo()
