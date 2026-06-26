"""Alternative to the sorted-scatter per-feature loss figure (Fig 2): a violin
plot of the per-feature MSE distribution at each loss exponent.

Shows the same point --- L2 spreads error unevenly (bimodal: half the features
near 0, half near 1/3), the higher exponents concentrate it --- but as a
distribution per exponent rather than a sorted curve. For comparison with the
current scatter so we can pick whichever reads better (feedback item A4).

  python plot_perfeature_violin.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

EXPS = ["2", "2.5", "3", "4", "6", "8"]


def _get_exp(d, e):
    """Look up an exponent's array, tolerating 'exp_2' vs 'exp_2.0' key styles."""
    for k in (f"exp_{e}", f"exp_{float(e)}", f"exp_{int(float(e))}"):
        if k in d.files:
            return d[k]
    raise KeyError(f"exponent {e} not in {d.files}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/seed_experiments.npz")
    ap.add_argument("--baseline", default="data/feedback_experiments.npz")
    ap.add_argument("--out", default="figures/per_feature_violin.png")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    # leftmost: the strongest non-superposition baseline (emulate-bias),
    # then the per-feature MSE pooled over seeds for each training exponent
    b = np.load(args.baseline, allow_pickle=True)
    data = [b["pf_bias"].reshape(-1)]
    labels = ["emulate-\nbias"]
    colors = ["#de2d26"]                       # baseline: mid red
    for e in EXPS:
        data.append(_get_exp(d, e).reshape(-1))  # (seeds*features,)
        labels.append(f"$L^{{{e}}}$")
        colors.append("#a50f15" if e == "2" else "#08519c")  # L2 naive red, rest blue

    pos = range(1, len(data) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    # medians
    meds = [np.median(x) for x in data]
    ax.scatter(pos, meds, color="k", s=14, zorder=3, label="median")

    ax.axhline(1 / 3, color="#fb6a4a", lw=1.4, ls=(0, (6, 4)),
               label="do-nothing baseline (1/3)")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"per-feature MSE on $x>0$ samples")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
