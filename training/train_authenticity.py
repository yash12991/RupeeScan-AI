import os
from pathlib import Path
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Configuration. Authenticity training is experimental and requires a separate,
# independently sourced real/counterfeit dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.getenv(
    "RUPEESCAN_AUTHENTICITY_DATASET", PROJECT_ROOT / "datasets" / "authenticity"
)).resolve()
MODEL_SAVE_PATH = Path(os.getenv(
    "RUPEESCAN_AUTHENTICITY_MODEL", PROJECT_ROOT / "backend" / "models" / "currency_authenticity_pytorch.pth"
)).resolve()
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

CLASSES = {
    0: "fake",
    1: "real"
}

# Custom Dataset
class BanknoteAuthenticityDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.file_paths[idx]).convert("RGB")
        except Exception as e:
            # Fallback to a blank image if loading fails
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def compute_metrics(y_true, y_pred, num_classes=2):
    """
    Computes confusion matrix, accuracy, precision, recall, and f1-score manually.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0
    
    precision = []
    recall = []
    f1 = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        fscore = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precision.append(prec)
        recall.append(rec)
        f1.append(fscore)
        
    return cm, accuracy, precision, recall, f1

def train_model():
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 2. Gather dataset files
    print(f"Scanning dataset directory: {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        return

    fake_dir = os.path.join(DATASET_DIR, "fake")
    real_dir = os.path.join(DATASET_DIR, "real")

    all_files = []
    all_labels = []

    # Load Fake note images (Label = 0)
    fake_count = 0
    if os.path.exists(fake_dir):
        for root, _, files in os.walk(fake_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, filename))
                    all_labels.append(0)
                    fake_count += 1
    print(f"Found {fake_count} FAKE note images.")

    # Load Real note images (Label = 1)
    real_count = 0
    if os.path.exists(real_dir):
        for root, _, files in os.walk(real_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, filename))
                    all_labels.append(1)
                    real_count += 1
    print(f"Found {real_count} REAL note images.")

    if not all_files:
        print("Error: No images found in dataset!")
        return

    total_images = len(all_files)
    print(f"Total dataset size: {total_images} images.")

    # 3. Train-val split (80% Train, 20% Val)
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)

    split_idx = int(0.8 * total_images)
    train_files = all_files[:split_idx]
    train_labels = all_labels[:split_idx]
    val_files = all_files[split_idx:]
    val_labels = all_labels[split_idx:]

    print(f"Train split size: {len(train_files)} images.")
    print(f"Validation split size: {len(val_files)} images.")

    # 4. Transforms & DataLoaders
    # Apply data augmentation to make the model extremely robust to different camera angles and lighting
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BanknoteAuthenticityDataset(train_files, train_labels, transform=train_transform)
    val_dataset = BanknoteAuthenticityDataset(val_files, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 5. Model Initialization
    print("Loading pre-trained ResNet18 model for binary classification...")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Adapt final fully connected layer for 2 classes (fake vs real)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Move model to device
    model = model.to(device)

    # 6. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Training Loop
    print("\nStarting Training Loop...")
    print("-------------------------")
    
    best_val_acc = 0.0
    torch.cuda.empty_cache()
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        print(f"Epoch {epoch+1}/{EPOCHS} started...")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        
        # Calculate validation metrics
        cm, val_accuracy, precisions, recalls, f1s = compute_metrics(y_true, y_pred)
        
        elapsed = time.time() - start_time
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {val_accuracy*100:.2f}%")
        print(f"Time Taken: {elapsed:.1f}s")
        print("-----------------------------\n")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            print(f"New best validation accuracy: {best_val_acc*100:.2f}%. Saving model...")
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print("--------------------------------------------------")
    
    # 8. Load best model to evaluate final metrics on test set
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading best model weights for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(targets.numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    cm, final_acc, precisions, recalls, f1s = compute_metrics(y_true, y_pred)
    
    print("\n================ FINAL EVALUATION REPORT ================")
    print(f"Overall Accuracy: {final_acc * 100:.2f}%")
    print("---------------------------------------------------------")
    print(f"{'Class Label':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 57)
    for i in range(2):
        class_name = CLASSES[i]
        print(f"{class_name:<15} | {precisions[i]*100:<9.2f}% | {recalls[i]*100:<9.2f}% | {f1s[i]*100:<9.2f}%")
    print("---------------------------------------------------------")
    print("Confusion Matrix:")
    print(f"{'':<15}{'Pred Fake':<12}{'Pred Real':<12}")
    print(f"Actual Fake    {cm[0, 0]:<12}{cm[0, 1]:<12}")
    print(f"Actual Real    {cm[1, 0]:<12}{cm[1, 1]:<12}")
    print("=========================================================")

if __name__ == "__main__":
    train_model()
