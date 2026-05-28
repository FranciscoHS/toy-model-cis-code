"""Two heatmaps of W_in (shape N x F = 50 x 100):

  (1) the trained 100k-step model
  (2) a 3-parameter birregular K=5 ansatz   W_in = a*M.T + b*(1-M.T)

The ansatz uses the same overlap-reduced birregular code as in
``sweep_K_codes.py`` (K=5 family) and the (a, b) values fit by that sweep
for seed 0 (read from ``data/sweep_K_codes.json``). Each heatmap is saved
as a separate PNG, with rows/columns seriated by hierarchical clustering
on Hamming distances so codeword structure is visible.
"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from codes import regular_code, reduce_overlap

F, N, K = 100, 50, 5
WEIGHTS = Path("weights/codeword_100f_50n_L4_100000steps.pt")
SWEEP = Path("data/sweep_K_codes.json")
OUT_TRAINED = Path("figures/W_in_trained_100k.png")
OUT_FIT = Path("figures/W_in_3param_birregular_k5.png")


def seriate(mask_FN):
    """Reorder rows and columns of a binary mask so codeword structure looks
    block-diagonal. Hierarchical clustering with optimal leaf ordering on
    Hamming distances.
    """
    Mf = mask_FN.astype(float)
    feat_link = linkage(pdist(Mf, metric="hamming"), method="average",
                        optimal_ordering=True)
    feat_order = leaves_list(feat_link)
    neur_link = linkage(pdist(Mf.T, metric="hamming"), method="average",
                        optimal_ordering=True)
    neur_order = leaves_list(neur_link)
    return feat_order, neur_order


def main():
    # trained W_in
    sd = torch.load(WEIGHTS, map_location="cpu")
    W_in_trained = sd["W_in"].numpy()   # (N, F)

    # 3-param birregular fit parameters from the K=5 seed-0 entry of the sweep
    with open(SWEEP) as f:
        res = json.load(f)
    a, b, c = res["regular"]["5"]["params"][0]
    print(f"3-param fit (K=5 birregular, seed 0): a={a:.4f}  b={b:.4f}  c={c:.4f}")

    # rebuild the same code the sweep used (regular_code seed=0, then
    # reduce_overlap with seed=10)
    M_fit_FN = reduce_overlap(regular_code(F, N, K, seed=0), 200_000, seed=10)
    W_in_fit = (a * M_fit_FN.T.astype(np.float32)
                + b * (1.0 - M_fit_FN.T.astype(np.float32)))

    M_trained_FN = (W_in_trained.T > 0)
    feat_t, neur_t = seriate(M_trained_FN)
    feat_f, neur_f = seriate(M_fit_FN)

    W_in_trained_sorted = W_in_trained[neur_t][:, feat_t]
    W_in_fit_sorted = W_in_fit[neur_f][:, feat_f]

    vmax = max(abs(W_in_trained).max(), abs(W_in_fit).max())
    vmin = -vmax

    def heatmap(W, title, out):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(W, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_xlabel(f"feature index (reordered, F={F})")
        ax.set_ylabel(f"neuron index (reordered, N={N})")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=r"$W_{\rm in}$")
        fig.tight_layout()
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")

    heatmap(W_in_trained_sorted,
            r"$W_{\rm in}$  trained 100k-step model  (rows/cols seriated)",
            OUT_TRAINED)
    heatmap(W_in_fit_sorted,
            rf"$W_{{\rm in}} = a\,M^\top + b\,(1-M^\top)$  "
            rf"(birregular K={K}, $a={a:.3f}$, $b={b:.3f}$, seriated)",
            OUT_FIT)

    print()
    print("on/off entry stats (each model uses its own codeword mask):")
    for tag, W, mask in [("trained", W_in_trained, M_trained_FN.T),
                         ("fit    ", W_in_fit,     M_fit_FN.T)]:
        on = W[mask]
        off = W[~mask]
        print(f"  {tag}:  on  mean={on.mean():+.4f} std={on.std():.4f}  "
              f"off mean={off.mean():+.4f} std={off.std():.4f}")


if __name__ == "__main__":
    main()
