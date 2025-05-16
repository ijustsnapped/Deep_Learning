#!/usr/bin/env python
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import multiprocessing

# Import model factory
from models import getModel

class ISICTestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        df['fname'] = df['image'].astype(str).str.replace(r'\.jpg$', '', regex=True)
        df['label'] = df[self.CLASS_NAMES].values.argmax(axis=1)
        self.data = df[['fname','label']].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(self.img_dir / f"{row.fname}.jpg").convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row.label)

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("config_dir", type=str, help="Directory containing YAML config files")
    parser.add_argument("exp_name", type=str, help="Experiment config name (YAML filename without .yaml)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config_dir) / f"{args.exp_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Paths and parameters
    TEST_CSV    = Path(cfg['TEST_CSV'])
    TEST_IMG    = Path(cfg['TEST_IMG'])
    BATCH_SIZE  = int(cfg.get('BATCH_SIZE', 128))
    NUM_WORKERS = int(cfg.get('NUM_WORKERS', 0))  # Use 0 for Windows
    DEVICE      = torch.device(cfg.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    CLASS_NAMES = cfg['CLASS_NAMES']
    NUM_CLASSES = int(cfg['NUM_CLASSES'])

    # Data transforms
    eval_transform = transforms.Compose([
        transforms.Resize(tuple(cfg.get('RESIZE',[256,256]))),
        transforms.CenterCrop(cfg.get('CROP_SIZE',224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.get('NORM_MEAN',[0.485,0.456,0.406]),
                             std=cfg.get('NORM_STD',[0.229,0.224,0.225]))
    ])

    # Dataset & DataLoader
    ISICTestDataset.CLASS_NAMES = CLASS_NAMES
    test_ds = ISICTestDataset(TEST_CSV, TEST_IMG, eval_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model_cfg = {'model_type': cfg['MODEL_TYPE'], 'numClasses': NUM_CLASSES}
    model = getModel(model_cfg)().to(DEVICE)

    # Load checkpoint
    ckpt_dir = Path(cfg.get('CHECKPOINT_DIR','checkpoints')) / args.exp_name
    best_path = ckpt_dir / 'best.pth'
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
    state = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Evaluation
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Evaluating'):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # Metrics
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    confmat = confusion_matrix(all_labels, all_preds)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", confmat)

    # Save results
    out_dir = Path(cfg.get('OUTPUT_DIR','outputs')) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'labels.npy', np.array(all_labels))
    np.save(out_dir / 'preds.npy', np.array(all_preds))
    print(f"Saved labels and predictions to {out_dir}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    evaluate()
