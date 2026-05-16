# Sample efficiency: area under the smoothed (window=50) learning curve, integrated over [0, 1,000,000] env steps. Higher = learns faster + ends higher.

| task | backbone | n_seeds | mean_auc | sem_auc | ci95_lo | ci95_hi |
|---|---|---|---|---|---|---|
| Easy | GRU | 5 | -478555.79 | 7970.23 | -491631.27 | -464361.93 |
| Easy | S3M / S4D | 5 | -499795.63 | 558.96 | -500749.02 | -498791.82 |
| Easy | S5 | 5 | -499958.17 | 532.72 | -500802.01 | -498950.81 |
| Easy | Transformer-XL | 1 | -500530.98 | — | — | — |
| Medium | GRU | 5 | -499949.49 | 216.85 | -500286.35 | -499584.66 |
| Medium | S3M / S4D | 5 | -498967.55 | 733.42 | -500215.24 | -497719.86 |
| Medium | S5 | 5 | -499854.51 | 451.50 | -500546.61 | -499035.13 |
| Medium | Transformer-XL | 1 | -500187.89 | — | — | — |
| Hard | GRU | 5 | -500359.07 | 217.55 | -500731.89 | -499986.25 |
| Hard | S3M / S4D | 5 | -499888.25 | 235.64 | -500274.66 | -499468.16 |
| Hard | S5 | 5 | -500166.99 | 292.97 | -500711.64 | -499662.92 |
| Hard | Transformer-XL | 1 | -499877.12 | — | — | — |
