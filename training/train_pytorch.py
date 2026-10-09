"""Train the seven-class denomination model from fixed CSV manifests."""
from __future__ import annotations
import argparse, csv, json, os, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "datasets" / "prepared_denomination"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "currency_classifier_candidate.pth"
CLASSES = {0:"10_rupees",1:"20_rupees",2:"50_rupees",3:"100_rupees",4:"200_rupees",5:"500_rupees",6:"2000_rupees"}
ImageFile.LOAD_TRUNCATED_IMAGES = False

class BanknoteDataset(Dataset):
    def __init__(self, rows, transform=None): self.rows, self.transform = rows, transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row["path"]) as source: image = source.convert("RGB")
        return (self.transform(image) if self.transform else image), int(row["class_id"])

def load_manifest(path, expected_split):
    if not path.is_file(): raise FileNotFoundError(f"Manifest not found: {path}. Run prepare_denomination_data.py first.")
    with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    if not rows: raise ValueError(f"Manifest is empty: {path}")
    missing = [r["path"] for r in rows if not Path(r["path"]).is_file()]
    if missing: raise FileNotFoundError(f"{len(missing)} images are missing; first: {missing[0]}")
    bad = [r for r in rows if r.get("split") != expected_split or int(r["class_id"]) not in CLASSES]
    if bad: raise ValueError(f"Invalid row in {path}: {bad[0]}")
    return rows

def validate_manifests(splits):
    paths = {n:{r["path"] for r in rows} for n,rows in splits.items()}
    groups = {n:{r["group_id"] for r in rows} for n,rows in splits.items()}
    for left,right in (("train","val"),("train","test"),("val","test")):
        if paths[left] & paths[right]: raise ValueError(f"Image leakage between {left} and {right}")
        if groups[left] & groups[right]: raise ValueError(f"Source-group leakage between {left} and {right}")
    for name,rows in splits.items():
        present = {int(r["class_id"]) for r in rows}
        if present != set(CLASSES): raise ValueError(f"{name} lacks classes: {sorted(set(CLASSES)-present)}")

def compute_metrics(y_true, y_pred, num_classes=7):
    cm = np.zeros((num_classes,num_classes), dtype=int)
    for truth,prediction in zip(y_true,y_pred): cm[truth,prediction] += 1
    accuracy = float(np.trace(cm)/cm.sum()) if cm.sum() else 0.0
    precision=[]; recall=[]; f1=[]
    for idx in range(num_classes):
        tp=cm[idx,idx]; fp=cm[:,idx].sum()-tp; fn=cm[idx,:].sum()-tp
        p=float(tp/(tp+fp)) if tp+fp else 0.0; r=float(tp/(tp+fn)) if tp+fn else 0.0
        precision.append(p); recall.append(r); f1.append(2*p*r/(p+r) if p+r else 0.0)
    return cm,accuracy,precision,recall,f1

def evaluate(model, loader, criterion, device):
    model.eval(); total_loss=0.0; truth=[]; predictions=[]
    with torch.no_grad():
        for inputs,targets in loader:
            inputs,targets=inputs.to(device),targets.to(device); outputs=model(inputs)
            total_loss += criterion(outputs,targets).item()*inputs.size(0)
            truth.extend(targets.cpu().tolist()); predictions.extend(outputs.argmax(1).cpu().tolist())
    return total_loss/len(loader.dataset), compute_metrics(truth,predictions)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest-dir",type=Path,default=Path(os.getenv("RUPEESCAN_DENOMINATION_MANIFESTS",DEFAULT_MANIFEST_DIR)))
    parser.add_argument("--model",type=Path,default=Path(os.getenv("RUPEESCAN_DENOMINATION_MODEL",DEFAULT_MODEL_PATH)))
    parser.add_argument("--epochs",type=int,default=20); parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--learning-rate",type=float,default=3e-4); parser.add_argument("--patience",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42); parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.patience)<1: raise SystemExit("epochs, batch-size, and patience must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    splits={n:load_manifest(args.manifest_dir/f"{n}.csv",n) for n in ("train","val","test")}
    validate_manifests(splits)
    print(f"Manifest directory: {args.manifest_dir.resolve()}")
    for name,rows in splits.items():
        counts=Counter(CLASSES[int(r["class_id"])] for r in rows); print(f"{name}: {len(rows)} images | {dict(sorted(counts.items()))}")
    if args.dry_run:
        print("Dry run passed: all classes exist and no image/source-group leakage was found."); return
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    if device.type=="cuda": print(f"Device name: {torch.cuda.get_device_name(0)}")
    normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([transforms.RandomRotation(8)],p=.5),transforms.RandomPerspective(distortion_scale=.08,p=.25),transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.1),transforms.ToTensor(),normalize])
    eval_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
    loaders={
        "train":DataLoader(BanknoteDataset(splits["train"],train_transform),batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=="cuda"),
        "val":DataLoader(BanknoteDataset(splits["val"],eval_transform),batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda"),
        "test":DataLoader(BanknoteDataset(splits["test"],eval_transform),batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")}
    class_counts=Counter(int(r["class_id"]) for r in splits["train"])
    weights=torch.tensor([len(splits["train"])/(7*class_counts[i]) for i in range(7)],dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT); model.fc=nn.Linear(model.fc.in_features,7); model.to(device)
    optimizer=optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",patience=2,factor=.3)
    best_macro_f1=-1.0; stale_epochs=0; args.model.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        started=time.time(); model.train(); loss_sum=0.0; correct=0
        for batch_idx,(inputs,targets) in enumerate(loaders["train"],1):
            inputs,targets=inputs.to(device),targets.to(device); optimizer.zero_grad(set_to_none=True)
            outputs=model(inputs); loss=criterion(outputs,targets); loss.backward(); optimizer.step()
            loss_sum+=loss.item()*inputs.size(0); correct+=(outputs.argmax(1)==targets).sum().item()
            if batch_idx%10==0 or batch_idx==len(loaders["train"]): print(f"Epoch {epoch}/{args.epochs} batch {batch_idx}/{len(loaders['train'])} loss={loss.item():.4f}")
        val_loss,val_metrics=evaluate(model,loaders["val"],criterion,device); macro_f1=float(np.mean(val_metrics[4])); scheduler.step(macro_f1)
        print(f"Epoch {epoch}: train_loss={loss_sum/len(splits['train']):.4f} train_acc={correct/len(splits['train']):.4f} val_loss={val_loss:.4f} val_acc={val_metrics[1]:.4f} val_macro_f1={macro_f1:.4f} time={time.time()-started:.1f}s")
        if macro_f1>best_macro_f1:
            best_macro_f1=macro_f1; stale_epochs=0; torch.save(model.state_dict(),args.model); print(f"Saved best checkpoint to {args.model}")
        else:
            stale_epochs+=1
            if stale_epochs>=args.patience: print("Early stopping"); break
    model.load_state_dict(torch.load(args.model,map_location=device,weights_only=True))
    test_loss,test_metrics=evaluate(model,loaders["test"],criterion,device)
    cm,accuracy,precision,recall,f1=test_metrics
    report={"test_loss":test_loss,"accuracy":accuracy,"macro_f1":float(np.mean(f1)),"classes":{CLASSES[i]:{"precision":precision[i],"recall":recall[i],"f1":f1[i]} for i in range(7)},"confusion_matrix":cm.tolist(),"seed":args.seed}
    report_path=args.model.with_suffix(".metrics.json"); report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Final held-out test accuracy={accuracy:.4f}, macro_f1={report['macro_f1']:.4f}"); print(f"Metrics written to {report_path}")

if __name__=="__main__": main()
