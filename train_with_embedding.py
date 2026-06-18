"""Sanity check: does the L^4 per-feature loss survive a Braun-et-al. embedding?

The baseline toy model is axis-aligned: features ARE the input coordinates,
    y = relu(x @ W_in.T) @ W_out.T            W_in:(N,F)  W_out:(F,N)

Braun et al. instead place each feature as a random near-orthogonal direction
in a d=1000 residual stream via a FIXED embedding W_E (unit-norm rows), and
read the output back out with the transpose W_E.T:
    r = x @ W_E                                W_E:(F,d), rows unit-norm, FIXED
    y = relu(r @ W_in.T) @ W_out.T @ W_E.T     W_in:(N,d)  W_out:(d,N)

The N<F bottleneck (the actual computation-in-superposition) is identical in
both; only the input/output basis differs. We train both with the SAME budget
and compare per-feature MSE, so the comparison is apples-to-apples and does not
depend on the shipped 100k-step checkpoints.

    python train_with_embedding.py --steps 100000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from small_models import generate_batch, random_embedding, effective_weights, DEVICE

F, N, P = 100, 50, 0.02
D_EMBED = 1000
LR = 0.01
BATCH = 8192


def train(loss_exp, steps, W_E=None, seed=0, data_seed=31337):
    """Train W_in/W_out (optionally through a fixed embedding W_E). Returns
    detached (W_in, W_out). If W_E is None the model is axis-aligned (d=F)."""
    torch.manual_seed(data_seed)
    x_tr, y_tr = generate_batch(800_000, F, P)
    x_ev, y_ev = generate_batch(200_000, F, P)

    d_in = F if W_E is None else W_E.shape[1]
    torch.manual_seed(seed)
    W_in = (torch.rand(N, d_in, device=DEVICE) * 0.2 - 0.1).requires_grad_()
    W_out = (torch.rand(d_in, N, device=DEVICE) * 0.3 - 0.15).requires_grad_()
    opt = torch.optim.Adam([W_in, W_out], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def forward(x):
        r = x if W_E is None else x @ W_E
        h = torch.relu(r @ W_in.T) @ W_out.T
        return h if W_E is None else h @ W_E.T

    tag = "axis-aligned" if W_E is None else f"embedded d={d_in}"
    log_every = max(1, steps // 10)
    print(f"training L{loss_exp} [{tag}], {steps} steps...")
    for step in range(steps):
        idx = torch.randint(0, x_tr.shape[0], (BATCH,), device=DEVICE)
        loss = ((forward(x_tr[idx]) - y_tr[idx]).abs() ** loss_exp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            with torch.no_grad():
                ev = ((forward(x_ev) - y_ev).abs() ** loss_exp).mean().item()
            print(f"  step {step + 1:>6}/{steps}  L{loss_exp} = {ev:.4e}", flush=True)
    return W_in.detach(), W_out.detach()


def per_feature_mse(W_in, W_out, W_E=None, n_batches=100, batch_size=2048,
                    seed=9999):
    """Per-feature MSE conditioned on the feature being active (matches
    small_models.evaluate_per_feature_mse, with the optional embedding)."""
    torch.manual_seed(seed)
    err = torch.zeros(F, device=DEVICE)
    cnt = torch.zeros(F, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = generate_batch(batch_size, F, P)
            r = x if W_E is None else x @ W_E
            h = torch.relu(r @ W_in.T) @ W_out.T
            yhat = h if W_E is None else h @ W_E.T
            active = (x > 0).float()
            err += ((yhat - y) ** 2 * active).sum(dim=0)
            cnt += active.sum(dim=0)
    return (err / cnt.clamp_min(1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loss-exp", type=int, default=4, choices=[2, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-seed", type=int, default=12345)
    args = ap.parse_args()
    print(f"device: {DEVICE}")

    W_E = random_embedding(F, D_EMBED, args.embed_seed)
    # quick orthogonality check on the fixed embedding
    G = (W_E @ W_E.T).cpu().numpy()
    off = G[~np.eye(F, dtype=bool)]
    print(f"W_E gram off-diagonal: mean|.|={np.abs(off).mean():.4f} "
          f"max|.|={np.abs(off).max():.4f}  (1/sqrt(d)={1/np.sqrt(D_EMBED):.4f})")

    Win_b, Wout_b = train(args.loss_exp, args.steps, W_E=None, seed=args.seed)
    Win_e, Wout_e = train(args.loss_exp, args.steps, W_E=W_E, seed=args.seed)

    # save the embedded checkpoint (raw + effective weights) as the canonical model
    Win_eff, Wout_eff = effective_weights(Win_e, Wout_e, W_E)
    ckpt = Path(f"weights/embedded_100f_50n_d{D_EMBED}_L{args.loss_exp}_{args.steps}steps.pt")
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"W_in": Win_e.cpu(), "W_out": Wout_e.cpu(), "W_E": W_E.cpu(),
                "W_in_eff": Win_eff.cpu(), "W_out_eff": Wout_eff.cpu()}, ckpt)
    print(f"saved -> {ckpt}")

    mse_b = per_feature_mse(Win_b, Wout_b, W_E=None)
    mse_e = per_feature_mse(Win_e, Wout_e, W_E=W_E)
    print(f"baseline  per-feature MSE: mean={mse_b.mean():.4e} max={mse_b.max():.4e}")
    print(f"embedded  per-feature MSE: mean={mse_e.mean():.4e} max={mse_e.max():.4e}")

    out = Path(f"data/embedding_check_L{args.loss_exp}_{args.steps}steps.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez(out, mse_baseline=mse_b, mse_embedded=mse_e,
             meta=json.dumps(dict(F=F, N=N, P=P, d_embed=D_EMBED,
                                  loss_exp=args.loss_exp, steps=args.steps,
                                  seed=args.seed, embed_seed=args.embed_seed)))
    print(f"saved -> {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(np.sort(mse_b)[::-1], "o", ms=4, color="#3070b8",
            label=f"axis-aligned (mean {mse_b.mean():.3e})")
    ax.plot(np.sort(mse_e)[::-1], "o", ms=4, color="#c8412c",
            label=f"Braun embedding d={D_EMBED} (mean {mse_e.mean():.3e})")
    ax.axhline(1/3, color="#444", lw=1.4, ls=(0, (6, 4)),
               label="MSE if model outputs 0")
    ax.set_xlabel("feature index (each sorted descending by its own MSE)")
    ax.set_ylabel(r"per-feature MSE on $x>0$ samples")
    ax.set_title(f"L{args.loss_exp}, {args.steps} steps: axis-aligned vs Braun embedding")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figout = Path(f"figures/embedding_check_L{args.loss_exp}_{args.steps}steps.png")
    figout.parent.mkdir(exist_ok=True)
    fig.savefig(figout, dpi=130)
    print(f"saved -> {figout}")


if __name__ == "__main__":
    main()
