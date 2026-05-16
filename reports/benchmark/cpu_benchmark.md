# DeepGuard CPU Benchmark

- device: `cpu`
- torch_threads: `8`
- image_size: `224`
- image_count: `9`
- video_count: `0`
- peak_cpu_percent: `100.0`
- peak_ram_percent: `94.4`

## Image Face Detection
| requested_backend | active_backend | count | mean_ms | p95_ms |
| --- | --- | ---: | ---: | ---: |
| insightface | insightface | 9 | 546.81 | 1633.07 |
| mtcnn | mtcnn | 9 | 54.34 | 132.26 |
| haar | haar | 9 | 47.55 | 238.72 |

## Image Model Throughput
| backend | tta | latency_ms_per_image | images_per_s | mean_tta_variants |
| --- | --- | ---: | ---: | ---: |
| insightface | false | 201.35 | 4.97 | 1.0 |
| insightface | adaptive | 205.49 | 4.87 | 1.1 |
| insightface | true | 259.71 | 3.85 | 2.0 |
| mtcnn | false | 119.44 | 8.37 | 1.0 |
| mtcnn | adaptive | 140.35 | 7.13 | 1.1 |
| mtcnn | true | 254.38 | 3.93 | 2.0 |
| haar | false | 117.77 | 8.49 | 1.0 |
| haar | adaptive | 161.65 | 6.19 | 1.1 |
| haar | true | 257.52 | 3.88 | 2.0 |

## Video Processing
| backend | count | mean_s | p95_s | total_frames | effective_fps | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| insightface | 0 | 0.00 | 0.00 | 0 | 0.00 | ok |
| mtcnn | 0 | 0.00 | 0.00 | 0 | 0.00 | ok |
| haar | 0 | 0.00 | 0.00 | 0 | 0.00 | ok |

## Acceptance Check
| backend | tta | image_latency_ms | image_status |
| --- | --- | ---: | --- |
| insightface | false | 201.35 | ok |
| insightface | adaptive | 205.49 | ok |
| insightface | true | 259.71 | ok |
| mtcnn | false | 119.44 | ok |
| mtcnn | adaptive | 140.35 | ok |
| mtcnn | true | 254.38 | ok |
| haar | false | 117.77 | ok |
| haar | adaptive | 161.65 | ok |
| haar | true | 257.52 | ok |

## CPU Notes
- Prefer `insightface` first. If its model package is not cached, first startup may download the ONNX package.
- `mtcnn` is usually the slowest CPU backend; keep it for accuracy checks, not high-throughput API serving.
- `haar` is fastest but less accurate and should be treated as a load-shedding fallback.
- CPU TTA uses original + horizontal flip only; `adaptive` runs TTA only for uncertain samples.
