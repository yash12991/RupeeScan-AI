import os
from pathlib import Path
import cv2
import numpy as np
import random

# Define Currency Classes
CLASSES = {
    0: {"name": "10_rupees", "color": (45, 82, 123)},      # Chocolate Brown (BGR)
    1: {"name": "20_rupees", "color": (0, 223, 218)},      # Greenish Yellow
    2: {"name": "50_rupees", "color": (250, 191, 0)},      # Fluorescent Blue
    3: {"name": "100_rupees", "color": (250, 230, 230)},   # Lavender
    4: {"name": "200_rupees", "color": (0, 223, 255)},     # Bright Yellow / Orange-Yellow
    5: {"name": "500_rupees", "color": (128, 128, 128)},   # Stone Grey
    6: {"name": "2000_rupees", "color": (127, 0, 255)}     # Magenta
}

def create_synthetic_banknote(denomination_id, width=300, height=140):
    """
    Creates a single synthetic Indian banknote image with characteristic colors and text.
    """
    # Create banknote base image
    note_info = CLASSES[denomination_id]
    note = np.zeros((height, width, 3), dtype=np.uint8)
    note[:] = note_info["color"]
    
    # Add border
    cv2.rectangle(note, (5, 5), (width - 6, height - 6), (255, 255, 255), 2)
    
    # Add some geometric patterns to look like a banknote
    cv2.circle(note, (40, height // 2), 25, (255, 255, 255), 1)
    cv2.rectangle(note, (width - 60, 20), (width - 20, height - 20), (255, 255, 255), 1)
    
    # Add currency text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(note, "RBI", (15, 25), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(note, f"Rs {note_info['name'].split('_')[0]}", (80, height // 2 + 10), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(note, "SPECIMEN", (width // 2 - 40, height - 15), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    return note

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_dataset(output_dir=None, num_train_per_class=30, num_val_per_class=10):
    """
    Generates a full synthetic dataset with images and annotations in YOLO format.
    """
    output_dir = Path(output_dir or PROJECT_ROOT / "data").resolve()
    print(f"Generating synthetic dataset in '{output_dir}' directory...")
    
    stages = ["train", "val"]
    for stage in stages:
        os.makedirs(os.path.join(output_dir, "images", stage), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", stage), exist_ok=True)
        
        num_imgs = num_train_per_class if stage == "train" else num_val_per_class
        
        for class_id, note_info in CLASSES.items():
            for i in range(num_imgs):
                # 1. Create a background image (random canvas size and color/texture)
                bg_w, bg_h = 640, 640
                background = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
                # Random background color (simulating different tabletops)
                bg_color = (random.randint(20, 100), random.randint(20, 100), random.randint(20, 100))
                background[:] = bg_color
                
                # Add some random noise/textures to the background
                for _ in range(10):
                    cx = random.randint(0, bg_w)
                    cy = random.randint(0, bg_h)
                    r = random.randint(20, 150)
                    cv2.circle(background, (cx, cy), r, (bg_color[0]+15, bg_color[1]+15, bg_color[2]+15), -1)
                
                # 2. Generate a banknote of random size and rotation
                scale = random.uniform(0.6, 1.2)
                note_w = int(300 * scale)
                note_h = int(140 * scale)
                note = create_synthetic_banknote(class_id, width=note_w, height=note_h)
                
                # 3. Rotate the banknote randomly
                angle = random.randint(-45, 45)
                center = (note_w // 2, note_h // 2)
                rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Calculate bounds of rotated image
                cos_val = np.abs(rot_matrix[0, 0])
                sin_val = np.abs(rot_matrix[0, 1])
                new_w = int((note_h * sin_val) + (note_w * cos_val))
                new_h = int((note_h * cos_val) + (note_w * sin_val))
                
                rot_matrix[0, 2] += (new_w / 2) - center[0]
                rot_matrix[1, 2] += (new_h / 2) - center[1]
                
                rotated_note = cv2.warpAffine(note, rot_matrix, (new_w, new_h), borderValue=(0, 0, 0))
                
                # 4. Place banknote onto background
                # Choose random placement
                if bg_w - new_w > 0:
                    px = random.randint(0, bg_w - new_w)
                else:
                    px = 0
                if bg_h - new_h > 0:
                    py = random.randint(0, bg_h - new_h)
                else:
                    py = 0
                
                # Mask to overlay rotated note
                gray_rot = cv2.cvtColor(rotated_note, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_rot, 1, 255, cv2.THRESH_BINARY)
                
                # Crop region of background
                roi = background[py:py+new_h, px:px+new_w]
                # Black-out the area of banknote in ROI
                bg_masked = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
                # Take only region of banknote from banknote image
                note_masked = cv2.bitwise_and(rotated_note, rotated_note, mask=mask)
                # Paste the note into the background
                dst = cv2.add(bg_masked, note_masked)
                background[py:py+new_h, px:px+new_w] = dst
                
                # 5. Add some image shadows or blurs for realism
                if random.random() > 0.5:
                    background = cv2.GaussianBlur(background, (5, 5), 0)
                
                # Add random brightness variation
                alpha = random.uniform(0.7, 1.3)
                background = np.clip(background * alpha, 0, 255).astype(np.uint8)
                
                # 6. Save image
                img_filename = f"{stage}_{note_info['name']}_{i}.jpg"
                img_path = os.path.join(output_dir, "images", stage, img_filename)
                cv2.imwrite(img_path, background)
                
                # 7. Compute bounding box coordinates (YOLO format: class_id x_center y_center width height)
                # Bounding box of the rotated note in absolute pixels
                x_center = px + (new_w / 2)
                y_center = py + (new_h / 2)
                w = new_w
                h = new_h
                
                # Normalize to [0, 1]
                x_center /= bg_w
                y_center /= bg_h
                w /= bg_w
                h /= bg_h
                
                label_filename = f"{stage}_{note_info['name']}_{i}.txt"
                label_path = os.path.join(output_dir, "labels", stage, label_filename)
                
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    
    # Write dataset.yaml configuration file for YOLO training
    yaml_content = """train: images/train
val: images/val

names:
"""
    for cid, info in CLASSES.items():
        yaml_content += f"  {cid}: {info['name']}\n"
        
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print("Dataset generation completed! Configuration written to:", yaml_path)

if __name__ == "__main__":
    generate_dataset()
