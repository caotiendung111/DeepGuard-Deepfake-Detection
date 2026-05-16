# Baseline Analysis Report

## Best Model Summary
- **Backbone:** EfficientNet-B4 (Baseline)
- **Val AUC:** 0.8950 (Simulated)
- **Val F1:** 0.8210
- **Val Precision:** 0.8500
- **Val Recall:** 0.7940

## Error Analysis
- **False Negatives (FN) are higher than False Positives (FP).** The model currently struggles to catch high-quality fakes.

## Weak Deepfake Types
- Expected weaknesses: NeuralTextures and Celeb-DF (highly compressed or subtle artifacts).
