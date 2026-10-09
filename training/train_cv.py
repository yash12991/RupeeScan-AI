import os
from pathlib import Path
import cv2
import numpy as np
import pickle
import random

# Define classes matching the dataset generator
CLASSES = {
    0: "10_rupees",
    1: "20_rupees",
    2: "50_rupees",
    3: "100_rupees",
    4: "200_rupees",
    5: "500_rupees",
    6: "2000_rupees"
}

# Mapping real dataset folder names to class IDs
FOLDER_TO_CLASS = {
    "Rs.10": 0,
    "Rs.20": 1,
    "Rs.50": 2,
    "Rs.100": 3,
    "Rs.200": 4,
    "Rs.500": 5,
    "Rs.2000": 6
}

def extract_color_histogram(img):
    """
    Extracts a normalized 2D HSV Color Histogram (H and S channels).
    HSV is invariant to illumination brightness changes to some extent.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # We use 32 bins for Hue and 32 bins for Saturation
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

def extract_orb_descriptors(img, max_features=500):
    """
    Extracts ORB descriptors from the image.
    ORB is a fast alternative to SIFT/SURF.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = Path(os.getenv(
    "RUPEESCAN_DENOMINATION_DATASET", PROJECT_ROOT / "datasets" / "denomination"
)).resolve()
DEFAULT_MODEL_PATH = Path(os.getenv(
    "RUPEESCAN_CLASSIC_MODEL", PROJECT_ROOT / "backend" / "models" / "currency_classifier.pkl"
)).resolve()


def train_classifier(dataset_dir=DEFAULT_DATASET_DIR, model_path=DEFAULT_MODEL_PATH, templates_per_class=None):
    """
    "Trains" the CV classifier by saving reference color histograms and ORB descriptors
    for each banknote denomination from the training set.
    """
    print(f"Training classic CV banknote classifier on dataset: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} not found.")
        return
        
    model_data = {
        "classes": CLASSES,
        "templates": []
    }
    
    # Iterate through each folder mapping to a class
    for folder_name, class_id in FOLDER_TO_CLASS.items():
        class_dir = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found. Skipping...")
            continue
            
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            print(f"Warning: No images found in {class_dir}. Skipping...")
            continue
            
        print(f"Class {CLASSES[class_id]} ({folder_name}): Found {len(img_files)} total images.")
        
        # Determine files to process
        if templates_per_class is not None:
            sampled_files = random.sample(img_files, min(templates_per_class, len(img_files)))
            print(f"  Selected {len(sampled_files)} sampled templates for training.")
        else:
            sampled_files = img_files
            print(f"  Selected ALL {len(sampled_files)} templates for training.")
            
        processed_count = 0
        for img_name in sampled_files:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: Failed to load image {img_name}")
                continue
                
            # Resize image keeping aspect ratio to max dimension 640
            h, w = img.shape[:2]
            max_dim = 640
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                
            # Extract features
            hist = extract_color_histogram(img)
            descriptors = extract_orb_descriptors(img)
            
            # Save template (only if we have descriptors or hist)
            model_data["templates"].append({
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "histogram": hist,
                # We convert descriptors to list for pickle serialization compatibility
                "descriptors": descriptors.tolist() if descriptors is not None else None
            })
            
            processed_count += 1
            if processed_count % 200 == 0:
                print(f"  Processed {processed_count}/{len(sampled_files)} images...")
            
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model dict
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"Model successfully saved with {len(model_data['templates'])} templates at: {model_path}")

if __name__ == "__main__":
    # Use random seed for reproducibility
    random.seed(42)
    train_classifier()
