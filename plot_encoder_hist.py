"""Histogram alternative to the sorted-encoder-columns figure (Fig 3).

Feedback (item 3): "fix the bins but measure each bin's height individually,
overlay many scatter points per bin, with the bin line as the mean. ~5-10 bins."

Interpretation here: fixed value-bins across the encoder entry range; for each of
the 100 feature columns we histogram its 50 entries into those bins, giving a
per-column count per bin. We scatter the 100 per-column counts at each bin
(x-jittered) and draw the mean-across-columns as the bin height (bars). This
shows the codeword structure as a bimodal distribution -- a tall off-code bin
near -0.02 and a few entries spread over the on-code range ~0.3-0.4 -- while the
scatter conveys column-to-column spread.

NOTE: the exact intended form is ambiguous in the feedback ("ask Stefan"); this
is one reasonable version, produced alongside the original plot_encoder_columns.

    python plot_encoder_hist.py --weights weights/embedded_100f_50n_d1000_L4_100000steps.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str,
                    default="weights/embedded_100f_50n_d1000_L4_100000steps.pt")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--logy", action="store_true",
                    help="log y-axis so the small on-code bins are visible")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sd = torch.load(args.weights, map_location="cpu")
    W = (sd["W_in_eff"] if "W_in_eff" in sd else sd["W_in"]).numpy()   # (N, F)
    N, F = W.shape
    fig_out = Path(args.out) if args.out else (
        Path("figures/encoder_hist_embedded.png") if "W_in_eff" in sd
        else Path("figures/encoder_hist.png"))

    lo, hi = W.min(), W.max()
    edges = np.linspace(lo, hi, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    # per-column histogram counts -> (F, bins)
    counts = np.stack([np.histogram(W[:, j], bins=edges)[0] for j in range(F)])
    mean_h = counts.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(centers, mean_h, width=width * 0.9, color="#3070b8", alpha=0.30,
           label="mean count across features", zorder=1)
    rng = np.random.default_rng(0)
    for b in range(args.bins):
        jit = rng.uniform(-width * 0.28, width * 0.28, size=F)
        ax.plot(centers[b] + jit, counts[:, b], "o", ms=2.5, color="#c8412c",
                alpha=0.25, zorder=2)
    ax.axvline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    if args.logy:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("encoder entry value")
    ax.set_ylabel("count per feature column (50 entries each)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig_out.parent.mkdir(exist_ok=True)
    fig.savefig(fig_out, dpi=130)
    print(f"bins (value): {np.round(centers, 3).tolist()}")
    print(f"mean height : {np.round(mean_h, 2).tolist()}")
    print(f"saved -> {fig_out}")


if __name__ == "__main__":
    main()
