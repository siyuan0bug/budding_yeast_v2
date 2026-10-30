# Larger batch memory probes

Budding; RTX A6000; synthetic input, 500 time steps, two BF16 mixed-precision hybrid_mse_rel + AdamW updates. No AL experiment.

| Model | Width | Batch | Allocated GiB | Reserved GiB |
|---|---:|---:|---:|---:|
| fno | 192 | 1024 | 6.28 | 8.10 |
| deeponet | 256 | 1024 | 2.32 | 2.60 |
| unet | 32 | 1024 | 1.90 | 2.12 |
| unet | 64 | 1024 | 3.13 | 3.97 |
