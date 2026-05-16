"""
DeepGuard — Dataset Downloader
Hỗ trợ: FaceForensics++, Celeb-DF v2, DFDC

Usage:
    python scripts/data/download_datasets.py --dataset celebdf --output-dir data/raw
    python scripts/data/download_datasets.py --dataset ff++ --output-dir data/raw --compression c23
    python scripts/data/download_datasets.py --dataset dfdc --output-dir data/raw
    python scripts/data/download_datasets.py --all --output-dir data/raw
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add("logs/download.log", rotation="10 MB", level="DEBUG")

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 8192  # bytes per download chunk

CELEBDF_FILES = {
    "List_of_testing_videos.txt": "https://github.com/yuezunli/celeb-deepfakeforensics/raw/master/List_of_testing_videos.txt",
}

DFDC_KAGGLE_DATASET = "c/deepfake-detection-challenge"

FF_SCRIPT_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/download-FaceForensics.py"


# ── Utility functions ──────────────────────────────────────────────────────────
def download_file(url: str, dest: Path, desc: str = "", resume: bool = True) -> bool:
    """Download a file with progress bar and optional resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    initial_pos = 0

    if resume and dest.exists():
        initial_pos = dest.stat().st_size
        headers["Range"] = f"bytes={initial_pos}-"
        logger.debug(f"Resuming download from byte {initial_pos:,}")

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=30)

        if resp.status_code == 416:
            logger.info(f"File already fully downloaded: {dest.name}")
            return True

        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) + initial_pos

        mode = "ab" if resume and initial_pos > 0 else "wb"
        with open(dest, mode) as f, tqdm(
            desc=desc or dest.name,
            total=total,
            initial=initial_pos,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=80,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.success(f"Downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    except requests.RequestException as e:
        logger.error(f"Download failed [{url}]: {e}")
        return False


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify file integrity with MD5 checksum."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    actual = md5.hexdigest()
    if actual == expected_md5:
        logger.success(f"MD5 OK: {filepath.name}")
        return True
    logger.error(f"MD5 mismatch for {filepath.name}: expected={expected_md5}, got={actual}")
    return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract zip or tar archive with progress."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path.name} → {dest_dir}")

    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.namelist()
                for member in tqdm(members, desc=f"Extracting {archive_path.name}", ncols=80):
                    zf.extract(member, dest_dir)
        elif archive_path.suffix in (".tar", ".gz", ".bz2", ".xz") or archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:*") as tf:
                members = tf.getmembers()
                for member in tqdm(members, desc=f"Extracting {archive_path.name}", ncols=80):
                    tf.extract(member, dest_dir)
        else:
            logger.warning(f"Unknown archive format: {archive_path.suffix}")
            return False

        logger.success(f"Extraction complete: {dest_dir}")
        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def organize_to_real_fake(src_dir: Path, real_dest: Path, fake_dest: Path):
    """
    Scan src_dir and move files to real/ or fake/ based on subfolder names.
    Folders named 'original', 'real', 'youtube' → real/
    Everything else → fake/
    """
    real_dest.mkdir(parents=True, exist_ok=True)
    fake_dest.mkdir(parents=True, exist_ok=True)

    REAL_KEYWORDS = {"original", "real", "youtube", "actors", "original_sequences"}
    FAKE_KEYWORDS = {"deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter",
                     "fake", "manipulated", "celeb-synthesis"}

    moved_real = moved_fake = 0

    for file_path in tqdm(list(src_dir.rglob("*")), desc="Organizing files", ncols=80):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png"}:
            continue

        # Determine real or fake from parent folder names
        parts = {p.lower() for p in file_path.parts}
        is_real = bool(parts & REAL_KEYWORDS)
        is_fake = bool(parts & FAKE_KEYWORDS)

        if is_real and not is_fake:
            dest = real_dest / file_path.name
            shutil.copy2(file_path, dest)
            moved_real += 1
        elif is_fake:
            dest = fake_dest / file_path.name
            shutil.copy2(file_path, dest)
            moved_fake += 1
        else:
            logger.debug(f"Unclassified file (skipped): {file_path}")

    logger.info(f"Organized → real: {moved_real:,} | fake: {moved_fake:,}")
    return moved_real, moved_fake


# ── FaceForensics++ ────────────────────────────────────────────────────────────
def download_ff_plus_plus(output_dir: Path, compression: str = "c23"):
    """
    Download FaceForensics++ using the official download script.
    Requires registration at: https://docs.google.com/forms/d/...

    Args:
        output_dir: Destination root directory.
        compression: 'c0' (raw), 'c23' (light), 'c40' (heavy)
    """
    logger.info("=" * 60)
    logger.info("FaceForensics++ Downloader")
    logger.info("=" * 60)

    ff_script = output_dir / "download-FaceForensics.py"
    ff_raw_dir = output_dir / "ff_plus_plus_raw"
    real_dest = output_dir / "real"
    fake_dest = output_dir / "fake"

    # Step 1: Download official FF++ script
    logger.info("Downloading official FF++ download script...")
    if not download_file(FF_SCRIPT_URL, ff_script, desc="FF++ download script"):
        logger.error("Failed to get FF++ download script.")
        logger.info("Manual download: https://github.com/ondyari/FaceForensics")
        return False

    # Step 2: Prompt for credentials (FF++ requires request approval)
    logger.warning("⚠️  FaceForensics++ requires REGISTRATION to download.")
    logger.warning("   Fill the form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform")
    logger.warning("   You will receive a link via email.")

    link = input("\n📧 Paste your FF++ download link (or press Enter to skip): ").strip()
    if not link:
        logger.warning("Skipping FF++ download. Set up manually later.")
        return False

    # Step 3: Run official download script
    ff_raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ff_script),
        str(ff_raw_dir),
        "-d", "all",
        "-c", compression,
        "-t", "videos",
        "--server", "EU",
    ]

    logger.info(f"Running FF++ downloader (compression={compression})...")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        env = os.environ.copy()
        env["FF_LINK"] = link
        result = subprocess.run(cmd, env=env, check=True)
        logger.success("FF++ download complete!")
    except subprocess.CalledProcessError as e:
        logger.error(f"FF++ download failed: {e}")
        return False

    # Step 4: Organize into real/fake
    logger.info("Organizing FF++ files into real/ and fake/ directories...")
    organize_to_real_fake(ff_raw_dir, real_dest, fake_dest)
    return True


# ── Celeb-DF v2 ───────────────────────────────────────────────────────────────
def download_celebdf(output_dir: Path):
    """
    Download Celeb-DF v2.
    Requires Google Form request: https://github.com/yuezunli/celeb-deepfakeforensics

    Size: ~2.7 GB | 590 real + 5,639 fake videos
    """
    logger.info("=" * 60)
    logger.info("Celeb-DF v2 Downloader")
    logger.info("=" * 60)

    real_dest = output_dir / "real"
    fake_dest = output_dir / "fake"
    celebdf_raw = output_dir / "celebdf_raw"

    logger.warning("⚠️  Celeb-DF v2 requires REGISTRATION.")
    logger.warning("   Request access: https://github.com/yuezunli/celeb-deepfakeforensics")
    logger.warning("   You will receive a Google Drive link.")

    gdrive_url = input("\n🔗 Paste Celeb-DF v2 Google Drive download URL (or Enter to skip): ").strip()
    if not gdrive_url:
        logger.warning("Skipping Celeb-DF download.")
        _show_celebdf_manual_instructions(output_dir)
        return False

    # Try gdown for Google Drive
    try:
        import gdown
        celebdf_raw.mkdir(parents=True, exist_ok=True)
        archive_path = celebdf_raw / "Celeb-DF-v2.zip"
        logger.info("Downloading via gdown...")
        gdown.download(gdrive_url, str(archive_path), quiet=False)

        if archive_path.exists():
            extract_archive(archive_path, celebdf_raw)
            _organize_celebdf(celebdf_raw, real_dest, fake_dest)
            return True
    except ImportError:
        logger.warning("gdown not installed. Run: pip install gdown")
    except Exception as e:
        logger.error(f"gdown download failed: {e}")

    # Fallback: manual instruction
    _show_celebdf_manual_instructions(output_dir)
    return False


def _show_celebdf_manual_instructions(output_dir: Path):
    logger.info("\n" + "=" * 60)
    logger.info("CELEB-DF MANUAL SETUP INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("1. Download from: https://github.com/yuezunli/celeb-deepfakeforensics")
    logger.info("2. Extract to any folder")
    logger.info("3. Run this script with --organize-only:")
    logger.info(f"   python scripts/data/download_datasets.py --organize-celebdf <extracted_folder> --output-dir {output_dir}")
    logger.info("=" * 60)


def _organize_celebdf(raw_dir: Path, real_dest: Path, fake_dest: Path):
    """Organize Celeb-DF v2 structure."""
    real_dest.mkdir(parents=True, exist_ok=True)
    fake_dest.mkdir(parents=True, exist_ok=True)

    # Celeb-DF v2 structure:
    # Celeb-real/ → real YouTube celebrity videos
    # Celeb-synthesis/ → deepfake videos
    # YouTube-real/ → real YouTube videos

    mappings = [
        ("Celeb-real", real_dest),
        ("YouTube-real", real_dest),
        ("Celeb-synthesis", fake_dest),
    ]

    total_real = total_fake = 0
    for folder_name, dest in mappings:
        src = raw_dir / folder_name
        if not src.exists():
            # Try recursive search
            matches = list(raw_dir.rglob(folder_name))
            if matches:
                src = matches[0]
            else:
                logger.warning(f"Folder not found: {folder_name}")
                continue

        files = list(src.rglob("*.mp4")) + list(src.rglob("*.avi"))
        for f in tqdm(files, desc=f"Copying {folder_name}", ncols=80):
            target = dest / f"{folder_name}_{f.name}"
            shutil.copy2(f, target)

        count = len(files)
        if dest == real_dest:
            total_real += count
        else:
            total_fake += count
        logger.info(f"  {folder_name}: {count} files → {dest.name}/")

    logger.success(f"Celeb-DF organized: real={total_real} | fake={total_fake}")


# ── DFDC ──────────────────────────────────────────────────────────────────────
def download_dfdc(output_dir: Path):
    """
    Download DFDC via Kaggle API.
    Requires: kaggle.json in ~/.kaggle/

    Size: ~470 GB (parts 0-49)
    """
    logger.info("=" * 60)
    logger.info("DFDC (DeepFake Detection Challenge) Downloader")
    logger.info("=" * 60)

    real_dest = output_dir / "real"
    fake_dest = output_dir / "fake"
    dfdc_raw = output_dir / "dfdc_raw"

    # Check kaggle installation
    try:
        import kaggle
        logger.success("Kaggle API found")
    except ImportError:
        logger.error("kaggle not installed. Run: pip install kaggle")
        logger.info("Also set up ~/.kaggle/kaggle.json with your API credentials.")
        logger.info("Get credentials at: https://www.kaggle.com/settings → API")
        return False

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(f"Kaggle credentials not found at {kaggle_json}")
        logger.info("1. Go to https://www.kaggle.com/settings")
        logger.info("2. Click 'Create New Token' under API section")
        logger.info(f"3. Place the downloaded kaggle.json at: {kaggle_json}")
        return False

    dfdc_raw.mkdir(parents=True, exist_ok=True)

    logger.warning("⚠️  DFDC is ~470 GB. This will take very long!")
    logger.info("   Downloading parts 0-2 only by default (adjust --dfdc-parts).")

    n_parts = int(input("How many parts to download? (0=all 50 parts, default=2): ").strip() or "2")

    try:
        cmd = [
            sys.executable, "-m", "kaggle",
            "competitions", "download",
            "-c", "deepfake-detection-challenge",
            "-p", str(dfdc_raw),
        ]
        if n_parts > 0:
            logger.info(f"Downloading {n_parts} part(s)...")
            # Download specific parts
            for i in range(min(n_parts, 50)):
                part_cmd = cmd + ["-f", f"dfdc_train_part_{i:02d}.zip"]
                logger.info(f"Downloading part {i}...")
                subprocess.run(part_cmd, check=True)
        else:
            subprocess.run(cmd, check=True)

        # Extract and organize
        for zip_file in sorted(dfdc_raw.glob("*.zip")):
            logger.info(f"Extracting {zip_file.name}...")
            extract_dir = dfdc_raw / zip_file.stem
            extract_archive(zip_file, extract_dir)

        _organize_dfdc(dfdc_raw, real_dest, fake_dest)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"DFDC download error: {e}")
        return False


def _organize_dfdc(raw_dir: Path, real_dest: Path, fake_dest: Path):
    """Organize DFDC using metadata.json files."""
    import json

    real_dest.mkdir(parents=True, exist_ok=True)
    fake_dest.mkdir(parents=True, exist_ok=True)

    real_count = fake_count = skip_count = 0

    for metadata_file in tqdm(list(raw_dir.rglob("metadata.json")), desc="Processing DFDC metadata", ncols=80):
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {metadata_file}: {e}")
            continue

        part_dir = metadata_file.parent

        for filename, info in metadata.items():
            label = info.get("label", "").upper()
            src = part_dir / filename

            if not src.exists():
                skip_count += 1
                continue

            if label == "REAL":
                dest = real_dest / f"dfdc_{src.stem}{src.suffix}"
                shutil.copy2(src, dest)
                real_count += 1
            elif label == "FAKE":
                dest = fake_dest / f"dfdc_{src.stem}{src.suffix}"
                shutil.copy2(src, dest)
                fake_count += 1

    logger.success(f"DFDC organized: real={real_count:,} | fake={fake_count:,} | skipped={skip_count:,}")


# ── Organize-only mode ─────────────────────────────────────────────────────────
def organize_existing(src_dir: Path, output_dir: Path, dataset_type: str = "auto"):
    """Organize an already-downloaded dataset into real/ fake/ structure."""
    real_dest = output_dir / "real"
    fake_dest = output_dir / "fake"

    if dataset_type == "celebdf":
        _organize_celebdf(src_dir, real_dest, fake_dest)
    elif dataset_type == "dfdc":
        _organize_dfdc(src_dir, real_dest, fake_dest)
    else:
        organize_to_real_fake(src_dir, real_dest, fake_dest)


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepGuard Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download Celeb-DF v2:
    python scripts/data/download_datasets.py --dataset celebdf --output-dir data/raw

  Download FaceForensics++ (c23 compression):
    python scripts/data/download_datasets.py --dataset ff++ --output-dir data/raw --compression c23

  Download DFDC (2 parts only):
    python scripts/data/download_datasets.py --dataset dfdc --output-dir data/raw

  Organize already-downloaded Celeb-DF:
    python scripts/data/download_datasets.py --organize-only /path/to/celebdf --output-dir data/raw --dataset celebdf
        """,
    )
    parser.add_argument("--dataset", choices=["ff++", "celebdf", "dfdc", "all"],
                        help="Dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output root directory (default: data/raw)")
    parser.add_argument("--compression", choices=["c0", "c23", "c40"], default="c23",
                        help="FF++ compression level (default: c23)")
    parser.add_argument("--organize-only", type=str, default=None,
                        help="Skip download, just organize existing folder into real/fake")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DeepGuard Dataset Downloader")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("=" * 60)

    # Organize-only mode
    if args.organize_only:
        src_dir = Path(args.organize_only)
        if not src_dir.exists():
            logger.error(f"Source directory does not exist: {src_dir}")
            sys.exit(1)
        organize_existing(src_dir, output_dir, dataset_type=args.dataset or "auto")
        return

    # Download mode
    results = {}
    datasets_to_download = []

    if args.all or args.dataset == "all":
        datasets_to_download = ["ff++", "celebdf", "dfdc"]
    elif args.dataset:
        datasets_to_download = [args.dataset]
    else:
        logger.error("Specify --dataset or --all")
        sys.exit(1)

    for dataset in datasets_to_download:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {dataset.upper()}")
        logger.info(f"{'=' * 60}")

        t0 = time.time()
        if dataset == "ff++":
            ok = download_ff_plus_plus(output_dir, compression=args.compression)
        elif dataset == "celebdf":
            ok = download_celebdf(output_dir)
        elif dataset == "dfdc":
            ok = download_dfdc(output_dir)
        else:
            ok = False

        elapsed = time.time() - t0
        results[dataset] = {"success": ok, "elapsed_sec": elapsed}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    for ds, result in results.items():
        status = "✅" if result["success"] else "❌"
        logger.info(f"  {status} {ds.upper():12s} — {result['elapsed_sec']:.1f}s")

    # Count final files
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    n_real = len(list(real_dir.glob("*"))) if real_dir.exists() else 0
    n_fake = len(list(fake_dir.glob("*"))) if fake_dir.exists() else 0
    logger.info(f"\nFinal count: real={n_real:,} | fake={n_fake:,}")
    logger.info(f"Output: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
