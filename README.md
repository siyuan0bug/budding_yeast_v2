# Budding Yeast Cell Cycle Simulation with Neural Operators

Neural operator-based surrogate modeling for budding yeast cell cycle dynamics, with active learning strategies for efficient data augmentation.

## Overview

This project builds surrogate models to predict the time-series trajectories of 39 molecular species in the budding yeast (*S. cerevisiae*) cell cycle, given a set of kinetic parameter perturbations (mutations). The core challenge is **sim-to-real transfer**: models are trained on ODE-simulated data but must generalize to real experimental mutant trajectories.

Key features:
- **Multiple neural operator architectures**: HyperFNO, CrossFNO, PureFNO, etc.
- **Active Learning (AL)**: Iteratively augment training data with the most informative samples, including the proposed **RGS (Real-Guided Sampling)** strategy
- **Physics validation**: All AL-generated candidates are verified by ODE solvers before being added to the training set
- **Wandb integration**: Real-time logging of training metrics, learning curves, and trajectory comparison plots

## Project Structure

```
budding_yeast_v2/
├── train.py                    # Training entry point
├── eval.py                     # Evaluation & visualization
├── analyze_seeds.py            # Multi-seed result analysis
├── run_pipeline.sh             # Batch experiment runner
├── models/
│   ├── yeast_lit_module.py     # PyTorch Lightning module
│   └── components/
│       ├── hyper_fno.py        # HyperFNO (hypernetwork + FNO + Neural ODE)
│       ├── cross_fno.py        # CrossFNO (cross-attention + FNO)
│       ├── pure_fno.py         # Pure FNO baseline
│       ├── hp_fno.py           # HP-FNO variant
│       ├── neural_ode.py       # Neural ODE baseline
│       └── ...                 # Other model variants
├── data/
│   ├── yeast_datamodule.py     # Data loading, splitting, normalization
│   ├── lhs_v1_origin.py        # LHS sampling engine + ODE simulation
│   ├── lhs_v1.py               # Alternative LHS implementation
│   └── dataset_utils.py        # Dataset utilities
└── utils/
    ├── al_callback.py          # Active learning callback (RGS, US, IS, etc.)
    ├── losses.py               # Loss functions (MSE, physics-informed)
    └── metrics.py              # Evaluation metrics (MAE, MSE, Correlation, etc.)
```

## Data

The dataset consists of:
- **~50,000 LHS-simulated samples**: Parameter perturbations sampled via Latin Hypercube Sampling, solved by `scipy.integrate.solve_ivp`
- **126 Real Mutant samples**: Experimentally validated mutant trajectories, used as the test set

Data format (`.npz`):
- `X`: Initial conditions, shape `(N, 39, 2)` — 39 variables, 2 features (value + derivative)
- `P`: Parameter vector, shape `(N, 141)` — 141 kinetic parameters
- `Y`: Time-series trajectories, shape `(N, 39, 500)` — 39 variables over 500 time steps (210 min)

## Models

| Model | Description | Key Feature |
|-------|-------------|-------------|
| `hyper_fno` | HyperFNO | Hypernetwork modulates FNO weights via parameter embedding |
| `cross_fno` | CrossFNO | Cross-attention between parameter embedding and FNO features |
| `pure_fno` | Pure FNO | Standard Fourier Neural Operator baseline |
| `hp_fno` | HP-FNO | Hybrid parameter-conditioned FNO |

## Active Learning Strategies

| Strategy | Code | Description |
|----------|------|-------------|
| **RGS** | `rgs` | Real-Guided Sampling: evaluates MAE on real mutants, allocates sampling quota to poorly-performing mutants, generates local dense samples around them |
| Uncertainty Sampling | `us` | Selects candidates with highest model uncertainty |
| Importance Sampling | `is` | Samples proportional to prediction error |
| Weighted Reservoir | `wrs` | Streaming-based weighted sampling |
| VeSSAL | `vessal` | Uncertainty + diversity balancing |
| HGGS | `hggs` | Manifold-based sampling |
| Random | `random` | Uniform random baseline |

### RGS Pipeline

```
1. Evaluate MAE on Real Mutant test set (denormalized)
2. Allocate sampling quota: 90% to poor mutants (MAE > threshold), 10% to global LHS
3. Generate local LHS samples within ±perturbation of mutant parameters
4. ODE-validate all candidates (physics consistency check)
5. Compute real MAE (model prediction vs ODE ground truth) for each candidate
6. Select top-k hardest samples (highest MAE) and add to training set
```

## Quick Start

### Installation

```bash
conda create -n neuralop python=3.10
conda activate neuralop
pip install torch pytorch-lightning wandb scipy numpy matplotlib
```

### Training

```bash
# Baseline (no active learning)
python train.py \
    --model hyper_fno \
    --loss_type mse_only \
    --dataset_name /path/to/dataset.npz \
    --modes 32 --width 32 \
    --batch_size 64 --max_epochs 300 \
    --lr 6e-4 --warmup_epochs 10 \
    --al_strategy none

# With RGS active learning
python train.py \
    --model hyper_fno \
    --loss_type mse_only \
    --dataset_name /path/to/dataset.npz \
    --modes 32 --width 32 \
    --batch_size 64 --max_epochs 300 \
    --lr 6e-4 --warmup_epochs 10 \
    --al_strategy rgs \
    --al_trigger_epoch 10 \
    --al_num_add 5000 \
    --al_perturbation 0.1 \
    --al_mae_threshold 0.1
```

### Batch Experiments

```bash
# Edit EXPERIMENTS array in run_pipeline.sh, then:
bash run_pipeline.sh
```

Each experiment line specifies 15 parameters:
```
MODEL LOSS_TYPE USE_ADJ DATASET_PATH DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD
```

### Evaluation

```bash
python eval.py \
    --ckpt /path/to/checkpoint.ckpt \
    --dataset_name /path/to/dataset.npz \
    --save_dir ./eval_result/experiment_name
```

Outputs per-mutant metrics (MAE, MSE, Relative L2, Correlation) and trajectory comparison plots.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 1e-3 | Initial learning rate (Cosine+Warmup schedule) |
| `--warmup_epochs` | 10 | Linear warmup epochs |
| `--batch_size` | 32 | Training batch size |
| `--modes` | 24 | Fourier modes in FNO layers |
| `--width` | 64 | Hidden channel width |
| `--al_strategy` | none | Active learning strategy |
| `--al_trigger_epoch` | 10 | AL trigger interval (epochs) |
| `--al_num_add` | 5000 | Samples added per AL iteration |
| `--al_perturbation` | 0.1 | RGS: local perturbation range (±10%) |
| `--al_mae_threshold` | 0.1 | RGS: MAE threshold for poor mutants |

## Learning Rate Schedule

Cosine Annealing with Linear Warmup:

```
lr
max ─┐    /────────\
     │   /          \
     │  /            \
     │ /              \
min ─┘/                \______
     0  warmup              max_epochs
```

## License

This project is for research purposes.
