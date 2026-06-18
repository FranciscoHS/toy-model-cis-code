"""Binned alternatives to the sorted-encoder-columns figure (Fig 3).

Feedback (item 3): "fix the bins but measure each bin's height individually,
overlay many scatter points per bin, with the bin line as the mean. ~5-10 bins."

mode=rank (default, the sensible reading): bin over RANK (sorted position), not
value. Each column's 50 entries are sorted descending; the 50 ranks are grouped
into ~8-10 bins. For each rank-bin we scatter the ACTUAL values across all 100
columns (so a point's vertical position is its value, not a count) and draw the
bin's mean value as a horizontal line ("height"). Off-code entries keep their
true ~-0.03 value -- nothing is collapsed to zero.

mode=value: a literal histogram over value bins (count per bin). Kept for
reference but collapses the off-code entries into a near-zero bin, which hides
the negative tail -- not recommended.

    python plot_encoder_hist.py --mode rank --bins 10
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_rank(W, nbins, fig_out):
    N, F = W.shape
    S = np.sort(W, axis=0)[::-1, :]                  # (N, F) descending per column
    edges = np.linspace(0, N, nbins + 1).astype(int)  # rank-bin boundaries

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rng = np.random.default_rng(0)
    means = []
    for b in range(nbins):
        r0, r1 = edges[b], edges[b + 1]
        vals = S[r0:r1, :].flatten()                 # actual values in this rank-bin
        xc = 0.5 * (r0 + r1)
        w = (r1 - r0)
        jit = rng.uniform(-w * 0.35, w * 0.35, size=vals.size)
        ax.plot(xc + jit, vals, "o", ms=2.0, color="#3070b8", alpha=0.12, zorder=1)
        m = vals.mean(); means.append((xc, w, m))
        ax.plot([xc - w * 0.45, xc + w * 0.45], [m, m], color="#c8412c", lw=2.4,
                zorder=3, solid_capstyle="round")
    ax.plot([], [], color="#c8412c", lw=2.4, label="bin mean")             # legend proxy
    ax.plot([], [], "o", ms=5, color="#3070b8", alpha=0.5, label="entry values")
    ax.axhline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel(f"entry rank in column (grouped into {nbins} bins)")
    ax.set_ylabel("encoder entry value")
    ax.set_xlim(0, N)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_out, dpi=130)
    print("rank-bin means (rank-center, width, mean value):")
    for xc, w, m in means:
        print(f"  ranks ~{xc:5.1f} (w={w}):  mean value = {m:+.4f}")
    print(f"saved -> {fig_out}")


def plot_value(W, nbins, fig_out, logy=False):
    N, F = W.shape
    edges = np.linspace(W.min(), W.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]); width = edges[1] - edges[0]
    counts = np.stack([np.histogram(W[:, j], bins=edges)[0] for j in range(F)])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(centers, counts.mean(0), width=width * 0.9, color="#3070b8", alpha=0.30,
           label="mean count across features")
    rng = np.random.default_rng(0)
    for b in range(nbins):
        jit = rng.uniform(-width * 0.28, width * 0.28, size=F)
        ax.plot(centers[b] + jit, counts[:, b], "o", ms=2.5, color="#c8412c", alpha=0.25)
    if logy:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.axvline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("encoder entry value"); ax.set_ylabel("count per feature column")
    ax.legend(loc="upper right", fontsize=10); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(fig_out, dpi=130)
    print(f"saved -> {fig_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str,
                    default="weights/embedded_100f_50n_d1000_L4_100000steps.pt")
    ap.add_argument("--mode", choices=["rank", "value"], default="rank")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--logy", action="store_true", help="value mode only")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sd = torch.load(args.weights, map_location="cpu")
    W = (sd["W_in_eff"] if "W_in_eff" in sd else sd["W_in"]).numpy()
    suffix = "_embedded" if "W_in_eff" in sd else ""
    fig_out = Path(args.out) if args.out else Path(
        f"figures/encoder_hist_{args.mode}{suffix}.png")
    fig_out.parent.mkdir(exist_ok=True)
    if args.mode == "rank":
        plot_rank(W, args.bins, fig_out)
    else:
        plot_value(W, args.bins, fig_out, logy=args.logy)


if __name__ == "__main__":
    main()
