"""
DeepGuard environment diagnostics.

This script is intentionally import-safe: it reports missing torch/cv2/etc.
instead of crashing before the useful checks run.
"""
import importlib
import importlib.metadata as metadata
import os
import sys
from pathlib import Path


REQUIRED_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "cv2": "opencv-python",
    "albumentations": "albumentations",
    "fastapi": "fastapi",
    "streamlit": "streamlit",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "yaml": "PyYAML",
}


def import_status(module_name: str):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return True, version or "installed"
    except Exception as exc:
        return False, str(exc)


def check_python():
    print("[INFO] Python executable:", sys.executable)
    print("[INFO] Python version   :", sys.version.split()[0])
    if sys.version_info >= (3, 14):
        print("[WARN] Python 3.14 may not be supported by all PyTorch wheels. Prefer Python 3.10-3.12.")

    venv_cfg = Path("venv311/pyvenv.cfg")
    if venv_cfg.exists():
        text = venv_cfg.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("executable = "):
                target = Path(line.split("=", 1)[1].strip())
                if not target.exists():
                    print(f"[WARN] venv311 points to missing interpreter: {target}")


def check_imports():
    print("\n[INFO] Checking importable runtime packages...")
    all_ok = True
    for module_name, package_name in REQUIRED_IMPORTS.items():
        ok, detail = import_status(module_name)
        if ok:
            print(f"  [OK] {module_name:<14} {detail}")
        else:
            all_ok = False
            print(f"  [FAIL] {module_name:<14} package={package_name} error={detail}")
    return all_ok


def check_cuda():
    print("\n[INFO] Checking CUDA...")
    ok, detail = import_status("torch")
    if not ok:
        print("[WARN] Cannot check CUDA because torch is not importable.")
        return

    import torch

    cuda_available = torch.cuda.is_available()
    print(f"[INFO] CUDA available: {cuda_available}")
    if cuda_available:
        print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"  Device {idx}: {torch.cuda.get_device_name(idx)}")


def check_requirements_file():
    print("\n[INFO] Checking requirements.txt parseability...")
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("[ERROR] requirements.txt not found.")
        return False

    bad_lines = []
    for line_no, line in enumerate(req_file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\x00" in stripped:
            bad_lines.append((line_no, "contains NUL byte"))
            continue
        package_name = stripped.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
        package_name = package_name.split("[", 1)[0].strip()
        try:
            metadata.version(package_name)
        except metadata.PackageNotFoundError:
            print(f"  [MISS] {stripped}")
        except Exception as exc:
            print(f"  [WARN] {stripped}: {exc}")

    if bad_lines:
        for line_no, reason in bad_lines:
            print(f"  [FAIL] line {line_no}: {reason}")
        return False

    print("  [OK] requirements.txt is readable.")
    return True


def check_project_files():
    print("\n[INFO] Checking project files...")
    checks = [
        Path("models/checkpoints/best_model.pth"),
        Path("configs/base.yaml"),
        Path("configs/thresholds.yaml"),
        Path("data/metadata/train.csv"),
        Path("data/metadata/val.csv"),
        Path("data/metadata/test.csv"),
    ]
    for path in checks:
        status = "OK" if path.exists() else "MISS"
        print(f"  [{status}] {path}")


def main() -> int:
    print("========================================")
    print("      DEEPGUARD ENVIRONMENT CHECK       ")
    print("========================================")
    check_python()
    imports_ok = check_imports()
    check_cuda()
    requirements_ok = check_requirements_file()
    check_project_files()
    print("========================================")
    return 0 if imports_ok and requirements_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
