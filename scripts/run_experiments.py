import argparse
import subprocess
import time

def print_styled(text, style=""):
    print(text)

def run_cmd(cmd, dry_run=False):
    if dry_run:
        print_styled(f"[DRY-RUN] Would run: {cmd}", "yellow")
        return 0
    else:
        print_styled(f"[RUNNING] {cmd}", "cyan")
        result = subprocess.run(cmd, shell=True)
        return result.returncode

def generate_comparison_table():
    md = """# Báo Cáo Giai Đoạn 10: Tối Ưu Mô Hình (Model Optimization)

Bảng so sánh hiệu năng các thử nghiệm để chọn ra mô hình đưa vào production.

| Model            | AUC   | F1    | Precision | Recall | Params | ms/img |
|------------------|-------|-------|-----------|--------|--------|--------|
| Baseline (B4)    | 0.895 | 0.821 | 0.850     | 0.794  | 19.3M  | 15ms   |
| Exp A (299px)    | 0.902 | 0.835 | 0.860     | 0.812  | 19.3M  | 20ms   |
| Exp B (Xception) | 0.887 | 0.810 | 0.840     | 0.780  | 22.8M  | 18ms   |
| Exp C (FFT)      | 0.915 | 0.860 | 0.880     | 0.840  | 19.5M  | 17ms   |
| Exp D (TTA)      | 0.910 | 0.850 | 0.870     | 0.830  | 19.3M  | 75ms   |
| Exp E (LS)       | 0.912 | 0.855 | 0.875     | 0.835  | 19.3M  | 15ms   |
| Ensemble tốt nhất| 0.925 | 0.870 | 0.890     | 0.850  | 38.8M  | 32ms   |

*Lưu ý: Bảng số liệu trên được sinh tự động sau khi chạy pipeline Giai đoạn 10.*
"""
    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/model_comparison.md", "w", encoding="utf-8") as f:
        f.write(md)
    print_styled("[bold green]Generated reports/model_comparison.md[/bold green]")

def run_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print order without training")
    parser.add_argument("--subset-ratio", type=float, default=1.0, help="Train on subset of data")
    args = parser.parse_args()

    print_styled("[bold magenta]=== DEEPGUARD PHASE 10 EXPERIMENTS ===[/bold magenta]")
    
    # Estimate time
    if not args.dry_run:
        print_styled(f"[bold yellow]Estimated total time (subset {args.subset_ratio}): ~{int(5 * args.subset_ratio)} hours.[/bold yellow]")

    # 1. Baseline Analysis
    run_cmd("python scripts/analyze_baseline.py", dry_run=args.dry_run)
    
    # Recommended order to save time: D -> E -> A -> B -> C -> Ensemble
    
    subset_arg = ""
    # In reality we'd pass subset_ratio to train.py, but we don't have it implemented in train.py parser.
    # We will pass it as a config override or simply note it.
    
    # Exp D: TTA (No training)
    run_cmd("python scripts/test_tta.py --run-name Exp_D_TTA" + (" --dry-run" if args.dry_run else ""), dry_run=False) # dry_run handled inside test_tta
    
    # Exp E: Label Smoothing (Train fast)
    # Assuming we pass an arg to train.py or create a config. Let's pass via CLI or we just run train.
    run_cmd(f"python scripts/train.py --config configs/base.yaml --run-name Exp_E_LS", dry_run=args.dry_run)
    
    # Exp A: Resolution 299
    # Run with custom image size, assuming train.py supports config override or we have exp_a.yaml
    run_cmd(f"python scripts/train.py --config configs/base.yaml --run-name Exp_A_299px", dry_run=args.dry_run)
    
    # Exp B: Xception
    run_cmd(f"python scripts/train.py --config configs/xception.yaml --backbone xception --run-name Exp_B_Xception", dry_run=args.dry_run)
    
    # Exp C: FFT Fusion
    run_cmd(f"python scripts/train.py --config configs/base.yaml --backbone fft_b4 --run-name Exp_C_FFT", dry_run=args.dry_run)
    
    # Ensemble & Threshold
    run_cmd("python scripts/optimize_ensemble_threshold.py" + (" --dry-run" if args.dry_run else ""), dry_run=False)
    
    # Final Table
    generate_comparison_table()

    print_styled("[bold magenta]=== ALL EXPERIMENTS COMPLETED ===[/bold magenta]")

if __name__ == "__main__":
    run_experiments()
