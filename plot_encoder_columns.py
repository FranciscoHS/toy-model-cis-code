"""Encoder-column profiles for the 100k-step trained model.

For each feature j, the encoder column W_in[:, j] is a vector of N=50
entries. We sort each column descending and overplot the 100 sorted
columns on the same axes. A median-across-features line is drawn on top.

The visible structure is a shared profile: a small number of large
positive entries (the on-codeword positions), then a flat near-zero tail
(the off-codeword positions). This grounds the binary-codeword schematic
in the actual trained weights without invoking any thresholding or
post-hoc concept.
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

WEIGHTS = Path("weights/codeword_100f_50n_L4_100000steps.pt")
FIG_OUT = Path("figures/encoder_columns_100k.png")


def main():
    sd = torch.load(WEIGHTS, map_location="cpu")
    W_in = sd["W_in"].numpy()                       # (N=50, F=100)
    N, F = W_in.shape
    sorted_cols = np.sort(W_in, axis=0)[::-1, :]    # descending, (N, F)

    print(f"W_in shape = {W_in.shape}")
    print(f"sorted column shape = {sorted_cols.shape}")
    print("per-rank summary (across 100 columns):")
    for r in [0, 1, 4, 5, 6, 9, 24, 49]:
        row = sorted_cols[r]
        print(f"  rank {r:2d}:  median = {np.median(row):+.4f}   "
              f"min = {row.min():+.4f}   max = {row.max():+.4f}")

    ranks = np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for j in range(F):
        ax.plot(ranks, sorted_cols[:, j], color="#3070b8", lw=0.7, alpha=0.18)
    median = np.median(sorted_cols, axis=1)
    ax.plot(ranks, median, color="#c8412c", lw=2.0,
            label="median across features")
    ax.axhline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("entry rank in column")
    ax.set_ylabel("entry value")
    ax.set_xlim(1, N)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_OUT, dpi=130)
    print(f"saved -> {FIG_OUT}")


if __name__ == "__main__":
    main()
