# toy-model-cis-code

Minimal code to reproduce the figures in the CiS toy-model paper.

## What's here

| Plot | Script | Output |
|---|---|---|
| Per-feature loss, L2 vs L4 (10 seeds, mean ± std) | `run_seed_experiments.py` + `plot_loss_per_feature_seeds.py` | `figures/loss_per_feature_seeds.png` |
| Per-feature-loss coefficient of variation vs loss exponent | `run_seed_experiments.py` + `plot_cv_vs_exponent.py` | CV numbers to stdout (figure optional) |
| L4 loss vs codeword length K (birregular vs random codes) | `sweep_K_codes.py` + `plot_K_sweep.py` | `figures/loss_vs_K_codes.png` |
| Encoder columns sorted descending (one curve per feature) | `plot_encoder_columns.py` | `figures/encoder_columns_100k.png` |

Plus:
- `run_seed_experiments.py` — train multiple seeds for each loss exponent in
  `{1, 2, 2.5, 3, 4, 6, 8}` (vectorized over seeds; seeds vary both init and
  data), then dump per-feature MSE to `data/seed_experiments.npz`. GPU recommended.
- `train_from_scratch.py` — train a single F=100, N=50, p=0.02 model for 100k steps (L2 or L4)
- `swap_test.py` — codeword-pattern swap test (function + CLI)

## Modules

- `small_models.py` — `SimpleMLP`, `generate_batch`, `DEVICE`
- `codes.py` — binary code generators (`regular_code`, `random_code`) and overlap-reduction edge swaps (`reduce_overlap`)

## Setup

```
pip install -r requirements.txt
```

## Reproducing the figures

The repo ships with the trained checkpoints (`weights/codeword_100f_50n_L4_100000steps.pt`,
`weights/codeword_100f_50n_L2_100000steps.pt`), the codeword sweep results
(`data/sweep_K_codes.json`), and the seed-sweep results
(`data/seed_experiments.npz`), so the plot scripts run immediately:

```
python plot_loss_per_feature_seeds.py
python plot_cv_vs_exponent.py
python plot_K_sweep.py
python plot_encoder_columns.py
```

To regenerate from scratch:

```
python run_seed_experiments.py --seeds 10   # populates data/seed_experiments.npz (GPU recommended)
python train_from_scratch.py --loss-exp 4   # single-network checkpoint; ~minutes on a GPU
python train_from_scratch.py --loss-exp 2
python sweep_K_codes.py                      # populates data/sweep_K_codes.json
```

## Swap test

```
python swap_test.py weights/codeword_100f_50n_L4_100000steps.pt
```
