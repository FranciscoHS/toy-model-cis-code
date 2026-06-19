"""Per-feature loss for L^2 vs L^4, aggregated over seeds.

Reads ``data/seed_experiments.npz`` (produced by run_seed_experiments.py).
For each model and each seed, the per-feature MSE is sorted descending; we
then aggregate across seeds at each rank and plot the mean curve with a +/-1
std band, plus faint per-seed curves underneath.
"""
import argparse
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=str(DATA))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--faint-exps", type=str, default="",
                    help="comma-separated extra exponents to overlay faintly, "
                         "e.g. 2.5,3,6,8")
    ap.add_argument("--baseline-npz", type=str, default=None,
                    help="feedback_experiments.npz; overlays the emulate-bias "
                         "per-feature curve as a baseline")
    args = ap.parse_args()
    data_path = Path(args.data)
    global FIG_OUT
    FIG_OUT = Path(args.out) if args.out else (
        Path("figures/per_feature_loss_embedded.png")
        if "embedded" in data_path.name else FIG_OUT)

    d = np.load(data_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))

    def get_exp(e):  # keys may be 'exp_2' or 'exp_2.0' depending on run
        for k in (f"exp_{e}", f"exp_{float(e)}", f"exp_{int(e)}"):
            if k in d.files:
                return d[k][:N_SEEDS]
        raise KeyError(f"exponent {e} not in {data_path.name}: {d.files}")

    mse_l2 = get_exp(2)
    mse_l4 = get_exp(4)
    F = mse_l2.shape[1]
    ranks = np.arange(F)

    _, mu2, sd2 = sorted_stats(mse_l2)
    _, mu4, sd4 = sorted_stats(mse_l4)
    print(f"L2 per-feature MSE: mean={mse_l2.mean():.4e}  max={mse_l2.max():.4e}")
    print(f"L4 per-feature MSE: mean={mse_l4.mean():.4e}  max={mse_l4.max():.4e}")

    # Two colour families so similar things read as similar at a glance:
    # reds = no-superposition group (L2/naive, emulate-bias, do-nothing baselines),
    # blues = superposition group (L3/L4/L6). L2 and L4 are the most saturated.
    c2, c4 = "#a50f15", "#08519c"          # L2 dark red, L4 dark blue (the two heroes)
    fig, ax = plt.subplots(figsize=(9, 4.6))

    # intermediate superposition exponents (L3, L6): lighter/mid shades of blue
    faint = [e.strip() for e in args.faint_exps.split(",") if e.strip()]
    exp_colors = ["#6baed6", "#3182bd", "#9ecae1"]   # blues (light, mid, lighter)
    for i, e in enumerate(faint):
        try:
            _, mu_e, _ = sorted_stats(get_exp(float(e) if "." in e else e))
        except KeyError as err:
            print(f"  (skipping faint exp {e}: {err})"); continue
        ax.plot(ranks, mu_e, "o", markersize=3, color=exp_colors[i % len(exp_colors)],
                alpha=0.8, zorder=1, label=rf"trained with L$^{{{e}}}$")

    # mean over seeds at each sorted rank, +/- std as error bars (dots, no lines)
    ax.errorbar(ranks, mu2, yerr=sd2, color=c2, ls="none", marker="o",
                markersize=4, alpha=0.85, elinewidth=1.0, capsize=0,
                label=r"trained with L$^2$")
    ax.errorbar(ranks, mu4, yerr=sd4, color=c4, ls="none", marker="o",
                markersize=4, alpha=0.85, elinewidth=1.0, capsize=0,
                label=r"trained with L$^4$")

    # do-nothing baseline 1/3 = E[x^2 | x ~ Uniform(0,1)]: light red, dashed
    ax.axhline(1.0 / 3.0, color="#fb6a4a", lw=1.6, ls=(0, (6, 4)), alpha=0.95,
               label=r"MSE on $x > 0$ samples if model outputs 0")

    # emulate-bias baseline (item 1): naive + optimal offset, mid red
    if args.baseline_npz:
        b = np.load(args.baseline_npz, allow_pickle=True)
        if "pf_bias" in b.files:
            ax.plot(ranks, np.sort(b["pf_bias"])[::-1], "o", markersize=4,
                    color="#de2d26", alpha=0.85, label="emulate-bias baseline")
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
