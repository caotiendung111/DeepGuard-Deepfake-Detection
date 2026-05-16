import os
import shutil
from pathlib import Path

def export_kaggle_outputs(project_dir="."):
    """
    Manually export DeepGuard results to Kaggle's root working directory.
    Run this in a Kaggle cell if the files are not showing up.
    """
    project_path = Path(project_dir)
    kaggle_root = Path("/kaggle/working")
    
    if not kaggle_root.exists():
        print("Not a Kaggle environment or /kaggle/working not found.")
        return

    # List of important files to export (Source Path -> Target Name)
    to_export = {
        project_path / "models/checkpoints/best_model.pth": "best_model.pth",
        project_path / "reports/evaluation/evaluation_report.pdf": "evaluation_report.pdf",
        project_path / "reports/evaluation/predictions.csv": "predictions.csv",
        project_path / "configs/thresholds.yaml": "final_thresholds.yaml",
    }

    print(f"Exporting results from {project_path.absolute()} to {kaggle_root}...")
    
    count = 0
    for src, name in to_export.items():
        if src.exists():
            dest = kaggle_root / name
            # Avoid self-copy if project_dir is already root
            if src.absolute() == dest.absolute():
                continue
                
            shutil.copy(src, dest)
            print(f"✅ Exported: {name}")
            count += 1
        else:
            print(f"❓ Not found: {src}")

    if count > 0:
        print(f"\nDone! {count} files exported. Refresh the Kaggle Output tab.")
    else:
        print("\nNo files found to export. Check if the paths are correct.")

if __name__ == "__main__":
    # If running as a script, assume we are inside the project folder
    export_kaggle_outputs()
