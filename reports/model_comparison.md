# Báo Cáo Giai Đoạn 10: Tối Ưu Mô Hình (Model Optimization)

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
