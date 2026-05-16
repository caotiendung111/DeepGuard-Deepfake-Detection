"""
DeepGuard — Dataset Classes
Hỗ trợ 2 chế độ: CSV-based và Folder-based.
Trả về (image_tensor, label, filepath) để tiện debug.
Tích hợp WeightedRandomSampler cho class imbalance.
"""
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ── Label constants ────────────────────────────────────────────────────────────
LABEL_MAP = {"real": 0, "fake": 1, "0": 0, "1": 1, 0: 0, 1: 1}
LABEL_NAMES = {0: "real", 1: "fake"}


# ═══════════════════════════════════════════════════════════════════════════════
# Folder-based Dataset (primary — scans data/faces/real/ + data/faces/fake/)
# ═══════════════════════════════════════════════════════════════════════════════
class DeepfakeFolderDataset(Dataset):
    """
    Load images directly from folder structure:
        data_root/
        ├── real/
        │   ├── vid001/frame_000001.jpg
        │   └── vid002/frame_000001.jpg
        └── fake/
            ├── vid003/frame_000001.jpg
            └── ...

    Returns:
        (image_tensor, label, filepath_str)

    Args:
        data_root: Root directory containing real/ and fake/ subdirs.
        transform: Albumentations Compose pipeline.
        allowed_extensions: Set of image file extensions to include.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
        allowed_extensions: Optional[set] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.extensions = allowed_extensions or self.EXTENSIONS

        self.samples: List[Tuple[str, int]] = []  # (filepath, label)
        self._scan_directory()

        if not self.samples:
            logger.warning(f"No images found in {self.data_root}. Check directory structure.")

    def _scan_directory(self):
        """Recursively scan for real and fake subdirectories with flexible naming."""
        found_any = False
        
        # Possible names for real and fake folders
        real_names = ["real", "Real", "REAL", "authentic", "Authentic"]
        fake_names = ["fake", "Fake", "FAKE", "deepfake", "Deepfake", "DEEPFAKE"]

        for label_name, label_int in [("real", 0), ("fake", 1)]:
            candidates = real_names if label_name == "real" else fake_names
            
            target_dir = None
            for cand in candidates:
                if (self.data_root / cand).exists():
                    target_dir = self.data_root / cand
                    break
            
            if target_dir:
                logger.info(f"Scanning {label_name} images in: {target_dir}")
                for ext in self.extensions:
                    for img_path in target_dir.rglob(f"*{ext}"):
                        self.samples.append((str(img_path), label_int))
                        found_any = True
            else:
                logger.warning(f"No folder found for {label_name} in {self.data_root} (tried {candidates})")

        if found_any:
            counts = self.get_class_counts()
            logger.info(f"Loaded {len(self.samples):,} images from {self.data_root} — Counts: {counts}")
        else:
            logger.error(f"No images found in {self.data_root} even with flexible naming!")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        filepath, label = self.samples[idx]

        # Load image as RGB numpy array
        try:
            image = Image.open(filepath).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}. Returning zeros.")
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            # Fallback: simple tensor conversion
            image_tensor = torch.from_numpy(
                image_np.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return image_tensor, torch.tensor(label, dtype=torch.long), filepath

    # ── Class distribution helpers ─────────────────────────────────────────────
    def get_labels(self) -> List[int]:
        """Return list of all labels (for sampler construction)."""
        return [label for _, label in self.samples]

    def get_class_counts(self) -> Dict[int, int]:
        """Return {label: count} dictionary."""
        return dict(Counter(self.get_labels()))

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights.
        Returns tensor of shape (num_classes,) — useful for loss functions.
        """
        counts = self.get_class_counts()
        total = sum(counts.values())
        # Inverse frequency: weight_i = total / (num_classes * count_i)
        num_classes = len(counts)
        weights = torch.tensor([
            total / (num_classes * counts.get(i, 1))
            for i in range(num_classes)
        ], dtype=torch.float32)
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weight for WeightedRandomSampler.
        Each sample gets the inverse-frequency weight of its class.
        """
        class_weights = self.get_class_weights()
        labels = self.get_labels()
        return torch.tensor([class_weights[label].item() for label in labels], dtype=torch.float64)

    def get_video_ids(self) -> List[str]:
        """Extract video_id from filepath (parent directory name)."""
        return [Path(fp).parent.name for fp, _ in self.samples]


# ═══════════════════════════════════════════════════════════════════════════════
# CSV-based Dataset (from split_dataset.py output)
# ═══════════════════════════════════════════════════════════════════════════════
class DeepfakeCSVDataset(Dataset):
    """
    Load images from a CSV file (output of split_dataset.py).

    Expected CSV columns: filepath, label [, label_name, video_id]

    Returns:
        (image_tensor, label, filepath_str)
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        data_root: Optional[str] = None,
        csv_file: Optional[str] = None,
    ):
        csv_path = csv_path or csv_file
        if csv_path is None:
            raise ValueError("Either csv_path or csv_file must be provided")

        self.data_root = Path(data_root) if data_root else None
        self.transform = transform

        self.df = pd.read_csv(csv_path, comment="#")
        assert "filepath" in self.df.columns, "CSV must have 'filepath' column"
        assert "label" in self.df.columns, "CSV must have 'label' column"

        self.df = self.df.dropna(subset=["filepath", "label"]).copy()
        self.df["filepath"] = self.df["filepath"].astype(str).str.strip()
        self.df = self.df[self.df["filepath"] != ""].copy()

        # Normalize labels
        self.df["label"] = self.df["label"].map(
            lambda x: LABEL_MAP.get(str(x).lower().strip(), x)
        )
        self.df["label"] = pd.to_numeric(self.df["label"], errors="coerce")
        invalid_rows = self.df["label"].isna().sum()
        if invalid_rows:
            logger.warning(f"Dropping {invalid_rows} rows with invalid labels from {csv_path}")
            self.df = self.df.dropna(subset=["label"]).copy()
        self.df["label"] = self.df["label"].astype(int)

        # Resolve paths
        if self.data_root:
            self.df["filepath"] = self.df["filepath"].apply(
                lambda p: str(self.data_root / p) if not Path(p).is_absolute() else p
            )

        logger.info(f"Loaded CSV: {csv_path} — {len(self.df):,} samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        filepath = str(row["filepath"])
        label = int(row["label"])

        try:
            image = Image.open(filepath).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(
                image_np.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return image_tensor, torch.tensor(label, dtype=torch.long), filepath

    def get_labels(self) -> List[int]:
        return self.df["label"].tolist()

    def get_class_counts(self) -> Dict[int, int]:
        return dict(Counter(self.get_labels()))

    def get_class_weights(self) -> torch.Tensor:
        counts = self.get_class_counts()
        total = sum(counts.values())
        num_classes = len(counts)
        return torch.tensor([
            total / (num_classes * counts.get(i, 1))
            for i in range(num_classes)
        ], dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        class_weights = self.get_class_weights()
        labels = self.get_labels()
        return torch.tensor([class_weights[label].item() for label in labels], dtype=torch.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Video Dataset (loads sequences of frames)
# ═══════════════════════════════════════════════════════════════════════════════
class DeepfakeVideoDataset(Dataset):
    """
    Dataset for video-based detection — loads N frames per video.

    Expected CSV: video_id, frame_dir, label, n_frames
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        n_frames: int = 16,
        data_root: Optional[str] = None,
    ):
        self.transform = transform
        self.n_frames = n_frames
        self.data_root = Path(data_root) if data_root else None
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        frame_dir = Path(row["frame_dir"])

        if self.data_root and not frame_dir.is_absolute():
            frame_dir = self.data_root / frame_dir

        frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        frame_paths = frame_paths[:self.n_frames]

        while len(frame_paths) < self.n_frames and frame_paths:
            frame_paths.append(frame_paths[-1])

        frames = []
        for fp in frame_paths:
            img = np.array(Image.open(fp).convert("RGB"))
            if self.transform:
                img = self.transform(image=img)["image"]
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            frames.append(img)

        video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return video_tensor, label, str(frame_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader Factory (with WeightedRandomSampler option)
# ═══════════════════════════════════════════════════════════════════════════════
def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    """
    Create DataLoader with optional WeightedRandomSampler for class balancing.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size.
        shuffle: Shuffle data (ignored if use_weighted_sampler=True).
        num_workers: Parallel data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        use_weighted_sampler: Use WeightedRandomSampler to handle class imbalance.
            When True, shuffle is automatically disabled (sampler handles it).

    Returns:
        DataLoader instance.
    """
    sampler = None

    if use_weighted_sampler and hasattr(dataset, "get_sample_weights"):
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # Sampler handles shuffling
        counts = dataset.get_class_counts()
        logger.info(
            f"WeightedRandomSampler enabled — "
            f"class counts: {counts} — "
            f"weights: real={sample_weights[0]:.3f}, fake={sample_weights[-1]:.3f}"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle or sampler is not None),
        persistent_workers=num_workers > 0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# High-level builder (used by train.py)
# ═══════════════════════════════════════════════════════════════════════════════
def build_datasets(
    config,
    train_transform=None,
    val_transform=None,
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """
    Build train/val/test datasets from config.

    Supports two modes:
    1. CSV-based: if config.train_csv exists → use DeepfakeCSVDataset
    2. Folder-based: scan data/faces/{split}/ directories

    Args:
        config: Config dataclass.
        train_transform: Augmentation for training.
        val_transform: Augmentation for validation/test.

    Returns:
        (train_dataset, val_dataset, test_dataset_or_None)
    """
    from .transforms import get_train_transforms, get_val_transforms

    if train_transform is None:
        train_transform = get_train_transforms(config.image_size)
    if val_transform is None:
        val_transform = get_val_transforms(config.image_size)

    # Try CSV mode first
    if Path(config.train_csv).exists() and Path(config.val_csv).exists():
        logger.info("Loading datasets from CSV files...")
        train_ds = DeepfakeCSVDataset(config.train_csv, transform=train_transform)
        val_ds = DeepfakeCSVDataset(config.val_csv, transform=val_transform)
        test_ds = None
        if hasattr(config, "test_csv") and Path(config.test_csv).exists():
            test_ds = DeepfakeCSVDataset(config.test_csv, transform=val_transform)
        return train_ds, val_ds, test_ds

    # Fallback: folder-based (check for 'faces' subfolder or direct real/fake structure)
    data_root = Path(config.data_root)
    
    # Check for split folders: Train, Validation, Test
    train_dir = data_root / "Train"
    val_dir = data_root / "Validation"
    test_dir = data_root / "Test"
    
    if train_dir.exists() and val_dir.exists():
        logger.info(f"Detected split structure: Train/Validation folders at {data_root}")
        train_ds = DeepfakeFolderDataset(str(train_dir), transform=train_transform)
        val_ds = DeepfakeFolderDataset(str(val_dir), transform=val_transform)
        test_ds = None
        if test_dir.exists():
            test_ds = DeepfakeFolderDataset(str(test_dir), transform=val_transform)
        return train_ds, val_ds, test_ds

    # Fallback to single folder
    faces_dir = data_root / "faces"
    target_dir = faces_dir if faces_dir.exists() else data_root
    
    if (target_dir / "real").exists() or (target_dir / "fake").exists():
        logger.info(f"Loading datasets from folder structure at {target_dir}...")
        train_ds = DeepfakeFolderDataset(str(target_dir), transform=train_transform)
        val_ds = DeepfakeFolderDataset(str(target_dir), transform=val_transform)
        logger.warning(
            "Using same folder for train and val. Run split_dataset.py for better validation!"
        )
        return train_ds, val_ds, None

    raise FileNotFoundError(
        f"No data found. Ensure CSV files exist or folder structure (real/fake) "
        f"exists at {target_dir}"
    )
