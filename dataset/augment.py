import cv2
import numpy as np
import os
import random

def augment_image(img, bbox=None):
    """
    Applies random augmentations to an image:
    - Random rotation (with bounding box recalculation if provided)
    - Random brightness and contrast adjustment
    - Random Gaussian blur (simulating camera motion blur)
    - Random shadows/illumination masks
    """
    h, w = img.shape[:2]
    augmented_img = img.copy()
    
    # 1. Random Brightness and Contrast
    alpha = random.uniform(0.7, 1.3)  # Contrast control
    beta = random.randint(-40, 40)    # Brightness control
    augmented_img = cv2.convertScaleAbs(augmented_img, alpha=alpha, beta=beta)
    
    # 2. Random Gaussian Blur
    if random.random() > 0.5:
        ksize = random.choice([3, 5, 7])
        augmented_img = cv2.GaussianBlur(augmented_img, (ksize, ksize), 0)
        
    # 3. Random Shadows
    if random.random() > 0.5:
        # Create a linear shadow gradient
        mask = np.ones((h, w), dtype=np.float32)
        p1 = (random.randint(0, w), random.randint(0, h))
        p2 = (random.randint(0, w), random.randint(0, h))
        
        # Simple gradient calculation
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        # Vector from p1 to p2
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
        v_len = np.linalg.norm(v)
        if v_len > 0:
            v /= v_len
            # Project each point onto the vector
            proj = (X - p1[0]) * v[0] + (Y - p1[1]) * v[1]
            proj_min, proj_max = proj.min(), proj.max()
            if proj_max - proj_min > 0:
                shadow_intensity = random.uniform(0.3, 0.7)
                norm_proj = (proj - proj_min) / (proj_max - proj_min)
                mask = 1.0 - (1.0 - shadow_intensity) * norm_proj
                mask = np.expand_dims(mask, axis=2)
                augmented_img = (augmented_img * mask).astype(np.uint8)

    # 4. Random Rotation (15 degrees max)
    if random.random() > 0.3:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        augmented_img = cv2.warpAffine(augmented_img, M, (w, h), borderValue=(0, 0, 0))
        
        # Adjust YOLO bounding box if provided
        if bbox is not None:
            # bbox is [class_id, x_center, y_center, bbox_w, bbox_h] in normalized coords
            cid, cx, cy, bw, bh = bbox
            # Bounding box corners in absolute coordinates
            cx_abs, cy_abs = cx * w, cy * h
            bw_abs, bh_abs = bw * w, bh * h
            
            # Simple approximation of rotated bounding box size
            rad = np.deg2rad(np.abs(angle))
            new_bw_abs = bw_abs * np.cos(rad) + bh_abs * np.sin(rad)
            new_bh_abs = bw_abs * np.sin(rad) + bh_abs * np.cos(rad)
            
            # Rotate center
            pt = np.array([cx_abs, cy_abs, 1.0])
            new_center = M.dot(pt)
            
            # Normalized values
            new_cx = new_center[0] / w
            new_cy = new_center[1] / h
            new_bw = new_bw_abs / w
            new_bh = new_bh_abs / h
            
            # Clip values to [0.0, 1.0]
            new_cx = np.clip(new_cx, 0.0, 1.0)
            new_cy = np.clip(new_cy, 0.0, 1.0)
            new_bw = np.clip(new_bw, 0.01, 1.0)
            new_bh = np.clip(new_bh, 0.01, 1.0)
            
            return augmented_img, [cid, new_cx, new_cy, new_bw, new_bh]

    return augmented_img, bbox

def augment_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir, num_aug_per_img=3):
    """
    Applies augmentations to all images in a directory and saves them.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Augmenting {len(img_files)} images from '{images_dir}'...")
    
    for img_name in img_files:
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_dir, base_name + '.txt')
        
        # Read YOLO label if exists
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        bboxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
        
        # Save original
        cv2.imwrite(os.path.join(output_images_dir, img_name), img)
        if bboxes:
            with open(os.path.join(output_labels_dir, base_name + '.txt'), 'w') as f:
                for bbox in bboxes:
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
        
        # Create augmented versions
        for a in range(num_aug_per_img):
            aug_img = img.copy()
            new_bboxes = []
            
            # Apply image-level augmentations and update bbox coords
            if bboxes:
                for bbox in bboxes:
                    aug_img, new_bbox = augment_image(aug_img, bbox)
                    if new_bbox:
                        new_bboxes.append(new_bbox)
            else:
                aug_img, _ = augment_image(aug_img)
            
            aug_name = f"{base_name}_aug_{a}"
            cv2.imwrite(os.path.join(output_images_dir, f"{aug_name}.jpg"), aug_img)
            
            if new_bboxes:
                with open(os.path.join(output_labels_dir, f"{aug_name}.txt"), 'w') as f:
                    for bbox in new_bboxes:
                        f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                        
    print("Dataset augmentation completed successfully!")

if __name__ == "__main__":
    # Example usage:
    # augment_dataset("data/images/train", "data/labels/train", "data/images/train_aug", "data/labels/train_aug")
    print("Augment.py module ready.")
