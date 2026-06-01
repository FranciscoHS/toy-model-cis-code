"""Per-feature loss for L^2 vs L^4, aggregated over seeds.

Reads ``data/seed_experiments.npz`` (produced by run_seed_experiments.py).
For each model and each seed, the per-feature MSE is sorted descending; we
then aggregate across seeds at each rank and plot the mean curve with a +/-1
std band, plus faint per-seed curves underneath.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA = Path("data/seed_experiments.npz")
FIG_OUT = Path("figures/loss_per_feature_seeds.png")
N_SEEDS = 10


def sorted_stats(mse):
    """mse:[S,F] -> (per-seed sorted [S,F], mean[F], std[F]) sorted descending."""
    s = np.sort(mse, axis=1)[:, ::-1]
    return s, s.mean(axis=0), s.std(axis=0, ddof=1)


def main():
    d = np.load(DATA, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    mse_l2 = d["exp_2"][:N_SEEDS]
    mse_l4 = d["exp_4"][:N_SEEDS]
    F = mse_l2.shape[1]
    ranks = np.arange(F)

    _, mu2, sd2 = sorted_stats(mse_l2)
    _, mu4, sd4 = sorted_stats(mse_l4)
    print(f"L2 per-feature MSE: mean={mse_l2.mean():.4e}  max={mse_l2.max():.4e}")
    print(f"L4 per-feature MSE: mean={mse_l4.mean():.4e}  max={mse_l4.max():.4e}")

    c2, c4 = "#c8412c", "#3070b8"
    fig, ax = plt.subplots(figsize=(9, 4.6))

    # mean over seeds at each sorted rank, +/- std as error bars (dots, no lines)
    ax.errorbar(ranks, mu2, yerr=sd2, color=c2, ls="none", marker="o",
                markersize=4, alpha=0.85, elinewidth=1.0, capsize=0,
                label=r"trained with L$^2$")
    ax.errorbar(ranks, mu4, yerr=sd4, color=c4, ls="none", marker="o",
                markersize=4, alpha=0.85, elinewidth=1.0, capsize=0,
                label=r"trained with L$^4$")

    # do-nothing baseline 1/3 = E[x^2 | x ~ Uniform(0,1)]
    ax.axhline(1.0 / 3.0, color="#444444", lw=1.6, ls=(0, (6, 4)), alpha=0.9,
               label=r"MSE on $x > 0$ samples if model outputs 0")
    ax.set_xlabel(r"feature index (each model sorted descending by its own per-feature MSE)")
    ax.set_ylabel(r"per-feature MSE on $x > 0$ samples")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_OUT, dpi=130)
    print(f"saved -> {FIG_OUT}")


if __name__ == "__main__":
    main()
