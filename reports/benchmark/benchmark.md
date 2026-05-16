# DeepGuard Benchmark

## Image Latency
| mode | count | errors | mean_s | p95_s | p99_s | rps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| false | 4 | 0 | 2.9612 | 3.3763 | 3.3763 | 0.67 |
| true | 4 | 0 | 2.7696 | 2.8273 | 2.8273 | 0.72 |
| adaptive | 2 | 2 | 2.6732 | 2.7633 | 2.7633 | 0.83 |

## Video Latency
| max_frames | count | errors | mean_s | p95_s | p99_s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 32 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 64 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |

## Outputs
- plots: `reports\benchmark\plots`
- resource samples: `33`
