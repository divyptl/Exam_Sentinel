"""
core/dataset.py

Dataset loaders for ExamSentinel training.

Supports:
  - FaceForensics++ (deepfake detection)
  - CASIA v2 (image splicing/forgery)
  - Screen-recapture dataset (moiré/recapture detection)
  - Custom exam frames (live capture)

Label format: multi-label binary vector
  [deepfake, recapture, spliced, forged]
   where 1 = manipulated, 0 = genuine
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Augmentation pipelines ──────────────────────────────────────────────────

def get_train_transforms(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ─── Dataset classes ──────────────────────────────────────────────────────────

class ExamSentinelDataset(Dataset):
    """
    Unified dataset that merges:
      1. FaceForensics++ frames  → deepfake label
      2. CASIA v2 images         → splicing/forgery label
      3. Recapture screen images → recapture label
      4. Genuine frames          → all zeros

    Each sample: (image_tensor, label_vector[4])
    label_vector: [is_deepfake, is_recapture, is_spliced, is_forged]
    """

    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 img_size: int = 256,
                 max_per_class: int = 5000):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.samples: List[Dict] = []

        self.transform = (get_train_transforms(img_size) if split == "train"
                          else get_val_transforms(img_size))

        self._load_faceforensics()
        self._load_casia()
        self._load_recapture()
        self._load_genuine()

        # Trim if needed
        if max_per_class and len(self.samples) > max_per_class * 4:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_per_class * 4]

        print(f"[Dataset] {split}: {len(self.samples)} samples loaded")
        self._print_stats()

    def _load_faceforensics(self):
        """
        FaceForensics++ structure:
        data/FaceForensics++/{real,fake}/{train,val,test}/*.png
        """
        ff_root = self.data_root / "FaceForensics++"
        if not ff_root.exists():
            print(f"  [!] FaceForensics++ not found at {ff_root}, skipping")
            return

        for label_name, is_fake in [("real", 0), ("fake", 1)]:
            folder = ff_root / label_name / self.split
            if not folder.exists():
                continue
            for img_path in folder.glob("*.png"):
                self.samples.append({
                    "path": str(img_path),
                    "labels": [is_fake, 0, 0, 0]  # [deepfake,recapture,spliced,forged]
                })

    def _load_casia(self):
        """
        CASIA v2 structure:
        data/CASIA_v2/{Au,Tp}/*.jpg
        Au = authentic, Tp = tampered
        """
        casia_root = self.data_root / "CASIA_v2"
        if not casia_root.exists():
            print(f"  [!] CASIA v2 not found at {casia_root}, skipping")
            return

        for label_name, spliced, forged in [("Au", 0, 0), ("Tp", 1, 1)]:
            folder = casia_root / label_name
            if not folder.exists():
                continue
            for ext in ["*.jpg", "*.png", "*.tif"]:
                for img_path in folder.glob(ext):
                    self.samples.append({
                        "path": str(img_path),
                        "labels": [0, 0, spliced, forged]
                    })

    def _load_recapture(self):
        """
        Recapture dataset structure:
        data/recapture/{genuine,recaptured}/*.jpg
        genuine = original digital image
        recaptured = photographed from screen (has moiré)
        """
        recap_root = self.data_root / "recapture"
        if not recap_root.exists():
            print(f"  [!] Recapture dataset not found at {recap_root}, skipping")
            return

        for label_name, is_recap in [("genuine", 0), ("recaptured", 1)]:
            folder = recap_root / label_name
            if not folder.exists():
                continue
            for ext in ["*.jpg", "*.png"]:
                for img_path in folder.glob(ext):
                    self.samples.append({
                        "path": str(img_path),
                        "labels": [0, is_recap, 0, 0]
                    })

    def _load_genuine(self):
        """
        Clean genuine exam frames (no manipulation).
        data/genuine/*.jpg — webcam captures of students in legitimate exams
        """
        genuine_root = self.data_root / "genuine"
        if not genuine_root.exists():
            return
        for ext in ["*.jpg", "*.png"]:
            for img_path in genuine_root.glob(ext):
                self.samples.append({
                    "path": str(img_path),
                    "labels": [0, 0, 0, 0]
                })

    def _print_stats(self):
        labels = np.array([s["labels"] for s in self.samples])
        names = ["deepfake", "recapture", "spliced", "forged"]
        for i, name in enumerate(names):
            pos = labels[:, i].sum()
            print(f"  {name:12s}: {int(pos):5d} positive / {len(self.samples):5d} total")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img = cv2.imread(sample["path"])
        if img is None:
            # Return blank on corrupt file
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        augmented = self.transform(image=img)
        img_tensor = augmented["image"]

        labels = torch.tensor(sample["labels"], dtype=torch.float32)
        return img_tensor, labels


class WebcamFrameDataset(Dataset):
    """
    Lightweight dataset for live inference on webcam frames.
    Used during demo — wraps raw numpy frames for batch processing.
    """

    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.transform = get_val_transforms(img_size)
        self.frames: List[np.ndarray] = []

    def add_frame(self, frame: np.ndarray):
        """Add a BGR OpenCV frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frames.append(rgb)

    def clear(self):
        self.frames.clear()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        aug = self.transform(image=self.frames[idx])
        return aug["image"]


# ─── Weighted sampler for imbalanced dataset ──────────────────────────────────

def build_weighted_sampler(dataset: ExamSentinelDataset) -> WeightedRandomSampler:
    """
    Handle class imbalance: genuine frames often outnumber tampered ones.
    Compute per-sample weights so each class is equally represented.
    """
    all_labels = np.array([s["labels"] for s in dataset.samples])

    # Use binary: genuine (all zeros) vs any manipulation
    is_manipulated = (all_labels.sum(axis=1) > 0).astype(int)

    class_counts = np.bincount(is_manipulated)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[is_manipulated]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(dataset),
        replacement=True
    )


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(data_root: str,
                    img_size: int = 256,
                    batch_size: int = 16,
                    num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    train_ds = ExamSentinelDataset(data_root, split="train", img_size=img_size)
    val_ds = ExamSentinelDataset(data_root, split="val", img_size=img_size)

    sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test with dummy data
    print("Testing dataset loader...")
    ds = ExamSentinelDataset("data", split="train")
    print(f"Total samples: {len(ds)}")
    if len(ds) > 0:
        img, lbl = ds[0]
        print(f"Image shape: {img.shape}, Labels: {lbl}")
    print("✓ Dataset loader OK")
