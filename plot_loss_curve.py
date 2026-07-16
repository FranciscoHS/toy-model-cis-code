"""Training-loss curve from a seed-experiment checkpoint.

Reads the ``loss_hist`` array that run_seed_experiments.py stores in each
--save-weights .pt (one scalar per 10 steps, the loss over all seeds jointly;
for exponents > 8 this is the p-norm mean(|err|^p)^(1/p), not mean(|err|^p)).

    python plot_loss_curve.py --weights weights/seedexp_100f_50n_L100_10seeds_100000steps.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    hist = np.asarray(sd["loss_hist"])
    every = int(sd.get("loss_hist_every", 10))
    exp = sd["loss_exp"]
    steps = np.arange(len(hist)) * every

    exp_tag = f"{exp:g}"
    out = Path(args.out) if args.out else Path(f"figures/loss_curve_L{exp_tag}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, hist, color="#3B6EF2", lw=0.6, alpha=0.25)
    win = 101
    if len(hist) > win:
        med = np.array([np.median(hist[max(0, i - win // 2):i + win // 2 + 1])
                        for i in range(len(hist))])
        ax.plot(steps, med, color="#3B6EF2", lw=1.8,
                label=f"rolling median ({win * every} steps)")
        ax.legend(frameon=False)
    ax.set_xlabel("training step")
    loss_name = (rf"$\mathrm{{mean}}(|err|^{{{exp_tag}}})^{{1/{exp_tag}}}$"
                 if exp > 8 else rf"$\mathrm{{mean}}(|err|^{{{exp_tag}}})$")
    ax.set_ylabel(f"batch loss  {loss_name}")
    ax.set_title(f"$L^{{{exp_tag}}}$ training loss "
                 f"({sd['seeds']} seeds jointly, logged every {every} steps)")
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"loss: start={hist[0]:.4f}  min={hist.min():.4f}  "
          f"final={hist[-1]:.4f}  (final 10%% mean={hist[int(.9 * len(hist)):].mean():.4f})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
