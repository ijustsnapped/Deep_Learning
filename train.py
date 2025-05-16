#!/usr/bin/env python
import os
import time
import random
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Auto-correct working directory to script's location
from pathlib import Path as _Path
os.chdir(_Path(__file__).resolve().parent)

# Import model factory
from models import getModel

# ----------------------------------------
# CLI argument parsing for config & experiment
# ----------------------------------------
parser = argparse.ArgumentParser(
    description="Train script for ISIC-2019 with configurable backbones"
)
parser.add_argument(
    "config_dir",
    type=str,
    help="Directory containing YAML config files"
)
parser.add_argument(
    "exp_name",
    type=str,
    help="Experiment config name (YAML filename without .yaml)"
)
args = parser.parse_args()

# Load YAML config
config_path = Path(args.config_dir) / f"{args.exp_name}.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# ----------------------------------------
# Configuration from YAML
# ----------------------------------------
# Data paths
DATA_ROOT   = Path(cfg['DATA_ROOT'])
TRAIN_CSV   = Path(cfg['TRAIN_CSV'])
TEST_CSV    = Path(cfg['TEST_CSV'])
TRAIN_IMG   = Path(cfg['TRAIN_IMG'])
TEST_IMG    = Path(cfg['TEST_IMG'])
for path in [TRAIN_CSV, TEST_CSV, TRAIN_IMG, TEST_IMG]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

# Hyperparameters
BATCH_SIZE     = int(cfg.get('BATCH_SIZE', 128))
NUM_EPOCHS     = int(cfg.get('NUM_EPOCHS', 40))
LEARNING_RATE  = float(cfg.get('LEARNING_RATE', 3e-4))
WEIGHT_DECAY   = float(cfg.get('WEIGHT_DECAY', 1e-4))
NUM_CLASSES    = int(cfg['NUM_CLASSES'])
TRAIN_FRACTION = float(cfg.get('TRAIN_FRACTION', 1.0))
NUM_WORKERS    = int(cfg.get('NUM_WORKERS', 4))
DEVICE         = torch.device(cfg.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
CLASS_NAMES    = cfg['CLASS_NAMES']

# Model configuration
model_cfg = {
    'model_type': cfg['MODEL_TYPE'],
    'numClasses': NUM_CLASSES
}

# ----------------------------------------
# Dataset Definition
# ----------------------------------------
class ISICDataset(Dataset):
    def __init__(self, csv_file: Path, img_dir: Path, transform=None):
        df = pd.read_csv(csv_file)
        df['fname'] = (
            df['image'].astype(str)
             .str.strip()
             .str.replace(r'\.jpg$', '', regex=True)
        )
        df['label'] = df[CLASS_NAMES].values.argmax(axis=1)
        self.data = df[['fname','label']].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img = Image.open(self.img_dir / f"{row.fname}.jpg").convert('RGB')
        if self.transform: img = self.transform(img)
        return img, int(row.label)

# ----------------------------------------
# Cutout Augmentation
# ----------------------------------------
class Cutout(nn.Module):
    def __init__(self, n_holes=1, length=16, p=0.5):
        super().__init__()
        self.n_holes = n_holes
        self.length = length
        self.p = p
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        C, H, W = img.shape
        for _ in range(self.n_holes):
            y = random.randint(0, H-1)
            x = random.randint(0, W-1)
            y1, y2 = max(0, y-self.length//2), min(H, y+self.length//2)
            x1, x2 = max(0, x-self.length//2), min(W, x+self.length//2)
            img[:, y1:y2, x1:x2] = 0
        return img

# ----------------------------------------
# Transform Pipeline
# ----------------------------------------
transform = transforms.Compose([
    transforms.Resize(tuple(cfg.get('RESIZE',[256,256]))),
    transforms.RandomCrop(cfg.get('CROP_SIZE',224)),
    transforms.RandomHorizontalFlip(p=cfg.get('HFLIP_P',0.5)),
    transforms.RandomVerticalFlip(p=cfg.get('VFLIP_P',0.5)),
    transforms.RandomApply([transforms.RandomAffine(
        degrees=cfg.get('AFFINE_DEGREES',45),
        scale=tuple(cfg.get('AFFINE_SCALE',[1.0,1.05]))
    )], p=cfg.get('AFFINE_P',0.5)),
    transforms.RandomApply([transforms.ColorJitter(*cfg.get('COLOR_JITTER',[0.2,0.2,0.2]))], p=cfg.get('CJ_P',0.5)),
    transforms.ToTensor(),
    Cutout(
        n_holes=cfg.get('CUTOUT_HOLES',1),
        length=cfg.get('CUTOUT_LENGTH',16),
        p=cfg.get('CUTOUT_P',0.5)
    ),
    transforms.Normalize(
        mean=cfg.get('NORM_MEAN',[0.485,0.456,0.406]),
        std=cfg.get('NORM_STD',[0.229,0.224,0.225])
    )
])

# ----------------------------------------
# Model, Loss, Optimizer, Scheduler
# ----------------------------------------
builder = getModel(model_cfg)
model = builder().to(DEVICE)

# Focal BCE loss
def focal_bce_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    if targets.dim()==1:
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float().to(inputs.device)
    prob = torch.sigmoid(inputs)
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob*targets + (1-prob)*(1-targets)
    loss = ((1-p_t)**gamma) * bce * alpha
    if reduction=='mean': return loss.mean()
    if reduction=='sum': return loss.sum()
    return loss

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# EMA setup
import copy
ema_model = copy.deepcopy(model).eval().to(DEVICE)
for p in ema_model.parameters(): p.requires_grad_(False)

def update_ema(ema, model, decay=0.9999):
    for ema_p, p in zip(ema.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1-decay)

# Checkpoint directory & best metric
ckpt_dir = Path(cfg.get('CHECKPOINT_DIR','checkpoints'))/args.exp_name
ckpt_dir.mkdir(parents=True, exist_ok=True)
best_acc = 0.0

# ----------------------------------------
# Training Loop
# ----------------------------------------
def train():
    full_train = ISICDataset(TRAIN_CSV, TRAIN_IMG, transform)
    train_size = int(TRAIN_FRACTION*len(full_train))
    train_ds, _ = random_split(full_train, [train_size, len(full_train)-train_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Experiment: {args.exp_name} | Training on {DEVICE} | epochs={NUM_EPOCHS} | bs={BATCH_SIZE}")
    global best_acc
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        epoch_loss=0.0; correct=0; total=0; start=time.time()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", ncols=80)
        for i,(imgs,labels) in enumerate(loop,1):
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad()
            logits=model(imgs)
            loss=focal_bce_loss(logits,labels)
            loss.backward(); optimizer.step()
            update_ema(ema_model,model)
            epoch_loss+=loss.item()
            preds=logits.argmax(dim=1)
            correct+=(preds==labels).sum().item(); total+=labels.size(0)
            loop.set_postfix(loss=f"{epoch_loss/i:.4f}", acc=f"{100*correct/total:.2f}%")
        elapsed=time.time()-start; m,s=divmod(elapsed,60)
        epoch_acc=100*correct/total; avg_loss=epoch_loss/len(train_loader)
        print(f"Epoch {epoch} done ({int(m)}m{int(s)}s) | Loss: {avg_loss:.4f} | Acc: {epoch_acc:.2f}%")

        # scheduler step
        scheduler.step()

        # checkpoint saving: last model every SAVE_FREQ epochs, best model on improvement
        save_freq = int(cfg.get('SAVE_FREQ', 5))
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), ckpt_dir / "last.pth")
            torch.save(ema_model.state_dict(), ckpt_dir / "ema_last.pth")
        # always update best when improved
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            torch.save(ema_model.state_dict(), ckpt_dir / "ema_best.pth")

if __name__=="__main__":
    train()
