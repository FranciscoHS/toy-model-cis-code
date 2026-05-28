# toy-model-cis-code

Minimal code to reproduce the figures in the CiS toy-model paper.

## What's here

| Plot | Script | Output |
|---|---|---|
| Per-feature loss, L2 vs L4 | `plot_loss_per_feature.py` | `figures/loss_per_feature_100k.png` |
| L4 loss vs codeword length K (birregular vs random codes) | `sweep_K_codes.py` + `plot_K_sweep.py` | `figures/loss_vs_K_codes.png` |
| Trained W_in heatmap + 3-parameter birregular fit | `plot_W_in_heatmaps.py` | `figures/W_in_trained_100k.png`, `figures/W_in_3param_birregular_k5.png` |

Plus:
- `train_from_scratch.py` — train the F=100, N=50, p=0.02 model for 100k steps (L2 or L4)
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
`weights/codeword_100f_50n_L2_100000steps.pt`) and the sweep results
(`data/sweep_K_codes.json`), so the plot scripts run immediately:

```
python plot_loss_per_feature.py
python plot_K_sweep.py
python plot_W_in_heatmaps.py
```

To regenerate from scratch:

```
python train_from_scratch.py --loss-exp 4   # ~minutes on a GPU
python train_from_scratch.py --loss-exp 2
python sweep_K_codes.py                     # populates data/sweep_K_codes.json
```

## Swap test

```
python swap_test.py weights/codeword_100f_50n_L4_100000steps.pt
```
