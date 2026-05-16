"""
Build train/val/test CSV metadata for the Kaggle dataset:
https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

The dataset is commonly extracted as:
  Dataset/
    Train/Real, Train/Fake
    Validation/Real, Validation/Fake
    Test/Real, Test/Fake

This script also tolerates lowercase names and flat real/fake folders. If
explicit Train/Validation/Test folders are not found, it creates a stratified
image-level split.
"""
import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_ALIASES = {
    "real": ("real", "reals", "authentic", "original"),
    "fake": ("fake", "fakes", "deepfake", "deepfakes", "generated"),
}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("validation", "valid", "val"),
    "test": ("test", "testing"),
}


@dataclass
class Record:
    filepath: str
    label: int
    label_name: str
    split: str


def iter_images(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def find_child(parent: Path, aliases: Iterable[str]) -> Path | None:
    if not parent.exists():
        return None
    alias_set = {alias.lower() for alias in aliases}
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() in alias_set:
            return child
    return None


def find_dataset_root(data_root: Path) -> Path:
    if any(find_child(data_root, aliases) for aliases in SPLIT_ALIASES.values()):
        return data_root
    nested = data_root / "Dataset"
    if nested.exists():
        return nested
    return data_root


def discover_explicit_splits(dataset_root: Path) -> List[Record]:
    records: List[Record] = []
    for split_name, split_aliases in SPLIT_ALIASES.items():
        split_dir = find_child(dataset_root, split_aliases)
        if split_dir is None:
            continue

        for label_name, label_aliases in LABEL_ALIASES.items():
            label_dir = find_child(split_dir, label_aliases)
            if label_dir is None:
                continue
            label_int = 0 if label_name == "real" else 1
            for image_path in iter_images(label_dir):
                records.append(Record(str(image_path), label_int, label_name, split_name))
    return records


def discover_flat_records(dataset_root: Path) -> List[tuple[Path, int, str]]:
    samples: List[tuple[Path, int, str]] = []
    for label_name, label_aliases in LABEL_ALIASES.items():
        label_dir = find_child(dataset_root, label_aliases)
        if label_dir is None:
            continue
        label_int = 0 if label_name == "real" else 1
        samples.extend((image_path, label_int, label_name) for image_path in iter_images(label_dir))
    return samples


def stratified_split(
    samples: List[tuple[Path, int, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> List[Record]:
    rng = random.Random(seed)
    records: List[Record] = []
    by_label: Dict[int, List[tuple[Path, int, str]]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample[1]].append(sample)

    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n = len(label_samples)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        split_samples = {
            "train": label_samples[:n_train],
            "val": label_samples[n_train:n_train + n_val],
            "test": label_samples[n_train + n_val:],
        }
        for split_name, split_items in split_samples.items():
            for image_path, label_int, label_name in split_items:
                records.append(Record(str(image_path), label_int, label_name, split_name))
    return records


def write_csvs(records: List[Record], output_dir: Path) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    fieldnames = ["filepath", "label", "label_name"]

    for split_name in ["train", "val", "test"]:
        split_records = [record for record in records if record.split == split_name]
        counts[split_name] = len(split_records)
        with open(output_dir / f"{split_name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in split_records:
                writer.writerow({key: getattr(record, key) for key in fieldnames})

    with open(output_dir / "all_splits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["split"])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Build metadata CSVs for Kaggle deepfake-and-real-images.")
    parser.add_argument("--data-root", default="data/raw/deepfake-and-real-images")
    parser.add_argument("--output-dir", default="data/metadata")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    dataset_root = find_dataset_root(data_root)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    records = discover_explicit_splits(dataset_root)
    mode = "explicit_split_folders"
    if not records:
        flat_samples = discover_flat_records(dataset_root)
        if not flat_samples:
            raise SystemExit(
                f"No images found under {dataset_root}. Expected Train/Real, Train/Fake "
                "or flat Real/Fake folders."
            )
        records = stratified_split(flat_samples, args.train, args.val, args.seed)
        mode = "stratified_image_split"

    output_dir = Path(args.output_dir)
    counts = write_csvs(records, output_dir)
    summary = {
        "dataset_root": str(dataset_root),
        "mode": mode,
        "counts": counts,
        "total": len(records),
        "output_dir": str(output_dir),
    }
    (output_dir / "kaggle_metadata_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
