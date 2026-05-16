# Steady-state throughput (median fps from metrics.jsonl, second half of training). Higher = faster wall-clock training.

| task | backbone | n_obs | median_fps | p25_fps | p75_fps |
|---|---|---|---|---|---|
| Easy | GRU | 500 | 55.7 | 54.8 | 87.4 |
| Easy | S3M / S4D | 500 | 53.3 | 52.1 | 68.4 |
| Easy | S5 | 500 | 102.1 | 100.8 | 103.2 |
| Easy | Transformer-XL | 100 | 12.2 | 12.1 | 12.3 |
| Medium | GRU | 500 | 87.8 | 87.0 | 89.3 |
| Medium | S3M / S4D | 500 | 57.7 | 56.9 | 57.9 |
| Medium | S5 | 500 | 89.9 | 78.5 | 90.6 |
| Medium | Transformer-XL | 100 | 11.7 | 11.7 | 11.8 |
| Hard | GRU | 500 | 74.9 | 70.2 | 75.3 |
| Hard | S3M / S4D | 500 | 58.1 | 57.9 | 58.8 |
| Hard | S5 | 500 | 78.9 | 78.3 | 89.5 |
| Hard | Transformer-XL | 100 | 11.8 | 11.7 | 11.9 |
