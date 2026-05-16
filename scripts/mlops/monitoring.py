"""
Script for monitoring model predictions in production.
Calculates data drift (shift in prediction distribution) and alerts if it exceeds threshold.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from pathlib import Path

LOGS_DIR = Path("logs")
# Let's say baseline confidence is ~ 0.5 across an even dataset.
# Or we compare the last 24 hours against the 7 days prior.
DRIFT_THRESHOLD = 0.10 # 10% shift

def check_model_drift():
    # Gather all prediction logs
    log_files = glob.glob(str(LOGS_DIR / "predictions*.csv"))
    if not log_files:
        print("No prediction logs found. Skipping monitoring.")
        return
        
    df = pd.concat((pd.read_csv(f) for f in log_files), ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    now = datetime.now()
    last_24h = df[df['timestamp'] > now - timedelta(days=1)]
    prev_7days = df[(df['timestamp'] <= now - timedelta(days=1)) & 
                    (df['timestamp'] > now - timedelta(days=8))]
                    
    print(f"--- MLOps Model Monitoring Report ({now.strftime('%Y-%m-%d %H:%M')}) ---")
    print(f"Total predictions last 24h: {len(last_24h)}")
    
    if len(last_24h) < 10:
        print("Not enough data in the last 24 hours for statistical comparison.")
        return
        
    current_mean_conf = last_24h['confidence'].mean()
    print(f"Current Mean Confidence (24h): {current_mean_conf:.3f}")
    
    if len(prev_7days) < 10:
        print("No baseline data from previous 7 days to compare against. Assuming 0.5 baseline.")
        baseline_conf = 0.5
    else:
        baseline_conf = prev_7days['confidence'].mean()
        print(f"Baseline Mean Confidence (7d): {baseline_conf:.3f}")
        
    shift = abs(current_mean_conf - baseline_conf)
    
    # Check if shift exceeds threshold
    if shift > DRIFT_THRESHOLD:
        print(f"⚠️ ALERT: Concept Drift Detected! Confidence shift is {shift*100:.1f}% (Threshold: {DRIFT_THRESHOLD*100}%)")
        # In a real system, send email / Slack alert here
    else:
        print("✅ Status Normal: No significant drift detected.")
        
    # Breakdown by input type
    print("\nBreakdown by Input Type (Last 24h):")
    if 'input_type' in last_24h.columns:
        print(last_24h.groupby('input_type')['confidence'].mean())
        
if __name__ == "__main__":
    check_model_drift()
