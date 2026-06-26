# toy-model-cis-code

Code to reproduce the figures and numbers in *Compressed Computation under L⁴
Loss is likely Computation in Superposition*.

## Paper figures → scripts

| Paper | Figure | How to produce |
|---|---|---|
| Fig 1 | `architecture.png` | hand-drawn schematic (no script) |
| Fig 2 | `per_feature_violin.png` | `python plot_perfeature_violin.py` |
| Fig 3 | `encoder_value_hist.png` | `python plot_encoder_hist.py --mode valuedots --weights weights/codeword_100f_50n_L4_100000steps.pt --out figures/encoder_value_hist.png` |
| Fig 4 | `encoder_code_hist.png` | `python plot_encoder_hist.py --mode kdist --axis both --weights weights/codeword_100f_50n_L4_100000steps.pt --out figures/encoder_code_hist.png` |
| Fig 5 | `swap.png` | schematic (no script); the 100% swap statistic comes from `swap_test.py` |
| Fig 6 | `mechanism.png` | hand-drawn schematic (no script) |
| Fig 7 | `synthetic_codes.png` | `python plot_K_sweep.py --noswap-data data/sweep_K_codes_noswap.json --out figures/synthetic_codes.png` |
| App. (attenuation) | `io_response.png` | `python plot_io_response.py` |

The embedded-model appendix figures (`*_embedded.png`) are produced by the same
plot scripts pointed at the embedded weights
(`weights/embedded_100f_50n_d1000_L4_100000steps.pt`) and the `*_embedded`
data files, with a matching `--out`.

## Numbers in the paper

- **Per-feature CV / loss spread, swap rate, decoder–pinv cosine, on/off-code
  values** — `recompute_mechanism_numbers.py` and `feedback_experiments.py`.
- **Impact table (§5)** — the rows come from `feedback_experiments.py`
  (baselines, free-per-entry, trained-decoder), `sweep_K_codes.py` (designed
  codes, with and without edge swaps), and `train_tied_pinv.py` (decoder tied
  to a pseudoinverse during training).
- **Swap test (§4)** — `python swap_test.py weights/codeword_100f_50n_L4_100000steps.pt`.

## Experiments / training

- `train_from_scratch.py` — train a single F=100, N=50, p=0.02 model for 100k
  steps (`--loss-exp 2` or `4`).
- `run_seed_experiments.py` — train multiple seeds for each loss exponent in
  `{1, 2, 2.5, 3, 4, 6, 8}` (vectorized over seeds), dump per-feature MSE to
  `data/seed_experiments.npz`. GPU recommended.
- `train_with_embedding.py` — Braun et al.'s embedded variant (fixed random
  `W_E` into ℝ¹⁰⁰⁰); verifies the CiS solution survives the embedding.
- `train_tied_pinv.py` — train with `W_out` tied to a scaled pseudoinverse of
  `W_in`, recomputed each step (isolates how much the freely-trained decoder
  buys over an exact pseudoinverse). → impact-table row.
- `train_dim.py` — does CiS survive other `(F, N, d_embed)` regimes? Tests a
  tight `d=50` bottleneck and a scaled-up `F=1000, d=500, N=500` config
  (Appendix: Other model sizes).
- `sweep_K_codes.py` — fit the 3-parameter ansatz to synthetic biregular/random
  codes across codeword length `K`; `--swaps 0` for the no-edge-swap variant.
  Writes `data/sweep_K_codes.json` (and `_noswap.json`).
- `biregular_value_variation.py` — do on-code encoder values still vary under a
  *designed* biregular code? (open-questions discussion in §5).
- `ablation_pinv_embedding.py` — transpose vs pseudoinverse unembedding
  (Appendix).

## Modules

- `small_models.py` — `SimpleMLP`, `generate_batch`, `DEVICE`.
- `codes.py` — binary code generators (`regular_code`, `random_code`) and the
  overlap-reduction edge swaps (`reduce_overlap`).

## Setup

```
pip install -r requirements.txt
```

## What ships in the repo

Trained checkpoints (`weights/codeword_100f_50n_L{2,4}_100000steps.pt`,
`weights/embedded_100f_50n_d1000_L4_100000steps.pt`,
`weights/tied_pinv_100f_50n_L4_100000steps.pt`), the seed sweep
(`data/seed_experiments.npz`), the codeword sweeps
(`data/sweep_K_codes.json`, `data/sweep_K_codes_noswap.json`), and the dim
sweeps (`data/dim_*.npz`), so the plot scripts above run without retraining.
