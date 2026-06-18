"""Coefficient of variation of per-feature loss vs training loss exponent.

Reads ``data/seed_experiments.npz``. For each loss exponent and seed we
compute the coefficient of variation (std/mean) of the per-feature MSE across
features, then plot the mean CV over seeds vs exponent with +/-1 std error bars.

A low CV means the loss is spread evenly across features (the superposition
solution); a high CV means a few features carry most of the error (the naive
solution). The point is that this varies smoothly with the exponent --- L^4 is
not a special value.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA = Path("data/seed_experiments.npz")
FIG_OUT = Path("figures/cv_vs_exponent.png")
N_SEEDS = 5
# We only show exponents that penalize outliers at least as much as L^2; L^1 is
# dropped because the claim is specifically about punishing outliers more than L^2.
MIN_EXP = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=str(DATA))
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    data_path = Path(args.data)
    fig_out = Path(args.out) if args.out else (
        Path("figures/cv_vs_exponent_embedded.png")
        if "embedded" in data_path.name else FIG_OUT)

    d = np.load(data_path, allow_pickle=True)
    json.loads(str(d["meta"]))  # validate the blob is present
    # npz keys look like 'exp_2', 'exp_2.5', ...
    keys = sorted((k for k in d.files if k.startswith("exp_")
                   and float(k[4:]) >= MIN_EXP),
                  key=lambda k: float(k[4:]))

    xs, mu, sd = [], [], []
    for k in keys:
        mse = d[k][:N_SEEDS]                              # [S, F]
        cv = mse.std(axis=1, ddof=0) / mse.mean(axis=1)  # [S]
        xs.append(float(k[4:]))
        mu.append(cv.mean())
        sd.append(cv.std(ddof=1))
        print(f"exp={k[4:]:<4}  CV mean={cv.mean():.3f}  std={cv.std(ddof=1):.3f}")
    xs, mu, sd = np.array(xs), np.array(mu), np.array(sd)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.errorbar(xs, mu, yerr=sd, color="#3070b8", ls="none", marker="o",
                markersize=8, capsize=4, zorder=3)

    ax.set_xlabel("training loss exponent")
    ax.set_ylabel("coefficient of variation of per-feature MSE")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])  # integers, but 2.5 stays 2.5
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig_out.parent.mkdir(exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    print(f"saved -> {fig_out}")


if __name__ == "__main__":
    main()
