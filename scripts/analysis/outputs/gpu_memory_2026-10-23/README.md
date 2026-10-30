# GPU memory measurements — 2026-10-23

RTX A6000, physical GPU 5; synthetic normalized inputs, batch512, 500 steps, BF16 mixed precision, hybrid_mse_rel loss, AdamW, clipping=1. Two complete forward/backward/update steps per isolated process. BioRefiner uses the actual training_step including coarse auxiliary loss. No W&B or full AL run.

| System | Model | Width | Blocks | Allocated peak GiB | Reserved peak GiB | Status |
|---|---|---:|---:|---:|---:|---|
| budding | fno | 192 | 4 | 3.19 | 4.10 | ok |
| budding | deeponet | 256 | 3 | 1.22 | 1.39 | ok |
| budding | unet | 32 | 4 | 0.97 | 1.09 | ok |
| budding | transformer | 192 | 4 | 8.81 | 10.69 | ok |
| budding | biom_refiner_plus | 192 | 4 | 32.33 | 34.65 | ok |
| budding | biom_refiner | 192 | 4 | 25.76 | 29.12 | ok |
| budding | hyper_fno | 192 | 4 | 23.23 | 24.59 | ok |
| fission | fno | 128 | 4 | 2.07 | 2.62 | ok |
| fission | biom_refiner_plus | 128 | 4 | 16.95 | 17.96 | ok |
| fission | graph_resnet_film | 128 | 4 | 8.27 | 8.78 | ok |

Allocated measures live PyTorch tensors; reserved includes PyTorch allocator cache. CUDA context, non-PyTorch allocations and other processes add GPU usage. These short-step peaks exclude validation, ensemble members, acquisition and full-run shape/allocator variation. GiB = 2^30 bytes.
