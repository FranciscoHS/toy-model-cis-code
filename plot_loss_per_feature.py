"""Per-feature loss scatter for L^2 vs L^4 trained models, F=100 N=50.

Loads ``weights/codeword_100f_50n_L{2,4}_100000steps.pt`` and plots
per-feature MSE on the same evaluation batches. Per-feature MSE is
conditioned on the feature being active:

    mean over samples where feature j is active of (model(x)_j - y_j)^2

If a checkpoint is missing, point ``train_from_scratch.py`` at the
corresponding loss exponent first.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from small_models import generate_batch, DEVICE, evaluate_per_feature_mse

F, N, P = 100, 50, 0.02
WEIGHTS_L4 = Path("weights/codeword_100f_50n_L4_100000steps.pt")
WEIGHTS_L2 = Path("weights/codeword_100f_50n_L2_100000steps.pt")
FIG_OUT = Path("figures/loss_per_feature_100k.png")


def load_weights(path):
    sd = torch.load(path, map_location=DEVICE)
    return sd["W_in"].to(DEVICE).float(), sd["W_out"].to(DEVICE).float()


def main():
    for p in (WEIGHTS_L4, WEIGHTS_L2):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}; run "
                f"`python train_from_scratch.py --loss-exp {p.stem.split('_L')[1][0]}`"
            )
    print(f"loading L4 weights from {WEIGHTS_L4}")
    Wi4, Wo4 = load_weights(WEIGHTS_L4)
    print(f"loading L2 weights from {WEIGHTS_L2}")
    Wi2, Wo2 = load_weights(WEIGHTS_L2)

    mse_l2 = evaluate_per_feature_mse(Wi2, Wo2, F, P)
    mse_l4 = evaluate_per_feature_mse(Wi4, Wo4, F, P)
    print(f"L2 per-feature MSE: mean={mse_l2.mean():.4e}  "
          f"median={np.median(mse_l2):.4e}  max={mse_l2.max():.4e}")
    print(f"L4 per-feature MSE: mean={mse_l4.mean():.4e}  "
          f"median={np.median(mse_l4):.4e}  max={mse_l4.max():.4e}")
    bad_l2 = (mse_l2 > 0.1).sum()
    bad_l4 = (mse_l4 > 0.1).sum()
    print(f"features with MSE > 0.1:  L2 = {bad_l2:>3}  L4 = {bad_l4:>3}")

    order = np.argsort(mse_l2)[::-1]
    ranks = np.arange(F)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.scatter(ranks, mse_l2[order], s=22, color="#c8412c", alpha=0.85,
               edgecolor="none", label=r"trained with L$^2$")
    ax.scatter(ranks, mse_l4[order], s=22, color="#3070b8", alpha=0.85,
               edgecolor="none", label=r"trained with L$^4$")
    # baseline 1/3 = E[x^2 | x ~ Uniform(0, 1)]: the per-feature MSE you would
    # incur on positive-x samples by outputting 0 instead of x.
    ax.axhline(1.0 / 3.0, color="#444444", lw=1.6, ls=(0, (6, 4)), alpha=0.9,
               label=r"MSE on $x > 0$ samples if model outputs 0")
    ax.set_xlabel(r"feature index (sorted descending by MSE for model trained with L$^2$)")
    ax.set_ylabel(r"per-feature MSE on $x > 0$ samples")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_OUT, dpi=130)
    print(f"saved -> {FIG_OUT}")


if __name__ == "__main__":
    main()
