"""Plot L4 loss vs codeword length K for birregular and random codes.

Reads results produced by ``sweep_K_codes.py``. Optionally overlays a second
(no-edge-swap) sweep as dashed lines, with no scatter points, to show the
effect of overlap minimization without cluttering the figure.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/sweep_K_codes.json")
OUT = Path("figures/loss_vs_K_codes.png")


def stats_for(res, K_list, l4_trained, code):
    means, stds, all_pts = [], [], []
    for K in K_list:
        ls = np.array(res[code][str(K)]["losses"]) / l4_trained
        means.append(ls.mean())
        stds.append(ls.std(ddof=1))
        all_pts.append(ls)
    return np.array(means), np.array(stds), all_pts


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=str(DATA))
    ap.add_argument("--noswap-data", type=str, default=None,
                    help="second sweep JSON (swaps=0) to overlay as dashed lines")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out)

    with open(args.data) as f:
        res = json.load(f)

    K_list = res["meta"]["K_list"]
    l4_trained = res["meta"]["l4_trained_100k"]

    reg_mu, reg_sd, reg_pts = stats_for(res, K_list, l4_trained, "regular")
    rnd_mu, rnd_sd, rnd_pts = stats_for(res, K_list, l4_trained, "random")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.errorbar(K_list, reg_mu, yerr=reg_sd, color="#1f77b4", lw=1.8,
                marker="o", markersize=7, capsize=4, label="Biregular (edge swaps)")
    ax.errorbar(K_list, rnd_mu, yerr=rnd_sd, color="#d62728", lw=1.8,
                marker="s", markersize=7, capsize=4, label="Random (edge swaps)")

    # optional no-edge-swap overlay: dashed lines, no scatter, to keep it readable
    if args.noswap_data:
        with open(args.noswap_data) as f:
            res_ns = json.load(f)
        reg_ns, _, _ = stats_for(res_ns, K_list, l4_trained, "regular")
        rnd_ns, _, _ = stats_for(res_ns, K_list, l4_trained, "random")
        ax.plot(K_list, reg_ns, color="#1f77b4", lw=1.6, ls="--",
                marker="o", markersize=5, mfc="none", label="Biregular (no swaps)")
        ax.plot(K_list, rnd_ns, color="#d62728", lw=1.6, ls="--",
                marker="s", markersize=5, mfc="none", label="Random (no swaps)")

    rng = np.random.default_rng(0)
    for K, pts in zip(K_list, reg_pts):
        jit = rng.uniform(-0.08, 0.08, size=len(pts))
        ax.plot(np.full_like(pts, K) + jit, pts, "o", color="#1f77b4",
                markersize=3.5, alpha=0.4)
    for K, pts in zip(K_list, rnd_pts):
        jit = rng.uniform(-0.08, 0.08, size=len(pts))
        ax.plot(np.full_like(pts, K) + jit, pts, "s", color="#d62728",
                markersize=3.5, alpha=0.4)

    ax.axhline(1.0, color="#666666", ls="--", lw=1.4,
               label="Trained model")

    ax.set_xticks(K_list)
    ax.set_xlabel("Codeword length K (active neurons per feature)")
    ax.set_ylabel(r"$L^4$ loss (ratio to trained model)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")
    print()
    print("summary (mean L4 +/- std over 5 seeds):")
    print(f"  {'K':>3}  {'regular':>20}  {'random':>20}  {'rand/reg':>9}")
    for K, rm, rs, nm, ns in zip(K_list, reg_mu, reg_sd, rnd_mu, rnd_sd):
        print(f"  {K:>3}  {rm:>10.3e} +/- {rs:>6.0e}  {nm:>10.3e} +/- {ns:>6.0e}  "
              f"{nm / rm:>8.2f}x")


if __name__ == "__main__":
    main()
