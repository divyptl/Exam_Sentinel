"""
core/dataset.py

Multi-source dataset loader for ExamSentinel training.
FaceForensics++ / CASIA v2 / Recapture / Genuine frames.
Labels: [deepfake, recapture, spliced, forged]
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _albu_version():
    return tuple(int(x) for x in A.__version__.split(".")[:2])


def get_train_transforms(img_size: int = 256) -> A.Compose:
    """Version-aware training transforms — handles albumentations v1.x and v2.x."""
    ver = _albu_version()
    base = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
    ]

    # ImageCompression API: quality_range in v1.4+, quality_lower/upper before
    try:
        if ver >= (1, 4):
            base.append(A.ImageCompression(quality_range=(60, 100), p=0.3))
        else:
            base.append(A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3))
    except Exception:
        pass

    # CoarseDropout API: changed in v2.0
    try:
        if ver >= (2, 0):
            base.append(A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.2
            ))
        else:
            base.append(A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2))
    except Exception:
        pass

    base += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(base)


def get_val_transforms(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class ExamSentinelDataset(Dataset):
    """
    Unified dataset. Labels: [deepfake, recapture, spliced, forged]
    Handles 0-sample case gracefully (no crash when data not downloaded yet).
    """

    def __init__(self, data_root: str, split: str = "train",
                 img_size: int = 256, max_per_class: int = 5000):
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

        print(f"[Dataset] {split}: {len(self.samples)} samples loaded")
        self._print_stats()

    def _load_faceforensics(self):
        ff_root = self.data_root / "FaceForensics++"
        if not ff_root.exists():
            return
        for label_name, is_fake in [("real", 0), ("fake", 1)]:
            folder = ff_root / label_name / self.split
            if not folder.exists():
                continue
            for img_path in folder.glob("*.png"):
                self.samples.append({"path": str(img_path), "labels": [is_fake, 0, 0, 0]})

    def _load_casia(self):
        casia_root = self.data_root / "CASIA_v2"
        if not casia_root.exists():
            return
        for label_name, spliced, forged in [("Au", 0, 0), ("Tp", 1, 1)]:
            folder = casia_root / label_name
            if not folder.exists():
                continue
            for ext in ["*.jpg", "*.png", "*.tif"]:
                for img_path in folder.glob(ext):
                    self.samples.append({"path": str(img_path), "labels": [0, 0, spliced, forged]})

    def _load_recapture(self):
        recap_root = self.data_root / "recapture"
        if not recap_root.exists():
            return
        for label_name, is_recap in [("genuine", 0), ("recaptured", 1)]:
            folder = recap_root / label_name
            if not folder.exists():
                continue
            for ext in ["*.jpg", "*.png"]:
                for img_path in folder.glob(ext):
                    self.samples.append({"path": str(img_path), "labels": [0, is_recap, 0, 0]})

    def _load_genuine(self):
        genuine_root = self.data_root / "genuine"
        if not genuine_root.exists():
            return
        for ext in ["*.jpg", "*.png"]:
            for img_path in genuine_root.glob(ext):
                self.samples.append({"path": str(img_path), "labels": [0, 0, 0, 0]})

    def _print_stats(self):
        # FIX: guard against empty dataset (0 samples → no crash)
        if len(self.samples) == 0:
            print("  (no samples — download datasets per data/README.md)")
            return
        labels = np.array([s["labels"] for s in self.samples])
        if labels.ndim == 1:
            labels = labels.reshape(-1, 4)
        names = ["deepfake", "recapture", "spliced", "forged"]
        for i, name in enumerate(names):
            pos = int(labels[:, i].sum())
            print(f"  {name:12s}: {pos:5d} pos / {len(self.samples):5d} total")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img = cv2.imread(sample["path"])
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=img)
        return augmented["image"], torch.tensor(sample["labels"], dtype=torch.float32)


class WebcamFrameDataset(Dataset):
    """Wraps live webcam frames for batch inference."""
    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.transform = get_val_transforms(img_size)
        self.frames: List[np.ndarray] = []

    def add_frame(self, frame: np.ndarray):
        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def clear(self):
        self.frames.clear()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(image=self.frames[idx])["image"]


def build_weighted_sampler(dataset: ExamSentinelDataset) -> WeightedRandomSampler:
    all_labels = np.array([s["labels"] for s in dataset.samples])
    is_manip = (all_labels.sum(axis=1) > 0).astype(int)
    counts = np.bincount(is_manip)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = weights[is_manip]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(dataset),
        replacement=True
    )


def get_dataloaders(data_root: str, img_size: int = 256,
                    batch_size: int = 16, num_workers: int = 4):
    train_ds = ExamSentinelDataset(data_root, split="train", img_size=img_size)
    val_ds   = ExamSentinelDataset(data_root, split="val",   img_size=img_size)
    sampler  = build_weighted_sampler(train_ds) if len(train_ds) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing dataset loader...")
    ds = ExamSentinelDataset("data", split="train")
    print(f"Total samples: {len(ds)}")
    print("✓ Dataset loader OK")
