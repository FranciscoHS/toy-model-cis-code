"""Does computation-in-superposition survive other (F, N, d_embed) regimes?

Stefan's appendix request: test (a) a <100-dim embedding of the standard
F=100, N=50 model (is the bottleneck too rough?), and (b) an order-of-magnitude
larger config, F=1000 features, d_embed=500 fixed random W_E, N=500 neurons.

Trains the embedded model under L^p, then reports the per-feature MSE spread.
The diagnostic for CiS is the coefficient of variation (CV = std/mean) of the
per-feature MSE: small (~0.03) => error spread evenly across all features
(superposition); large (~1) => naive solution (half learned, half ignored).

  python train_dim.py --F 100 --N 50 --d-embed 50
  python train_dim.py --F 1000 --N 500 --d-embed 500 --steps 100000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from small_models import generate_batch, random_embedding, DEVICE

LR = 0.01
BATCH = 8192


def train(F, N, P, d_embed, loss_exp, steps, W_E, seed=0, data_seed=31337):
    torch.manual_seed(data_seed)
    x_tr, y_tr = generate_batch(800_000, F, P)
    x_ev, y_ev = generate_batch(200_000, F, P)

    torch.manual_seed(seed)
    W_in = (torch.rand(N, d_embed, device=DEVICE) * 0.2 - 0.1).requires_grad_()
    W_out = (torch.rand(d_embed, N, device=DEVICE) * 0.3 - 0.15).requires_grad_()
    opt = torch.optim.Adam([W_in, W_out], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def forward(x):
        r = x @ W_E
        h = torch.relu(r @ W_in.T) @ W_out.T
        return h @ W_E.T

    log_every = max(1, steps // 10)
    print(f"training L{loss_exp} [F={F} N={N} d={d_embed}], {steps} steps...", flush=True)
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


def per_feature_mse(W_in, W_out, W_E, F, P, n_batches=100, batch_size=2048, seed=9999):
    torch.manual_seed(seed)
    err = torch.zeros(F, device=DEVICE)
    cnt = torch.zeros(F, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = generate_batch(batch_size, F, P)
            r = x @ W_E
            yhat = (torch.relu(r @ W_in.T) @ W_out.T) @ W_E.T
            active = (x > 0).float()
            err += ((yhat - y) ** 2 * active).sum(dim=0)
            cnt += active.sum(dim=0)
    return (err / cnt.clamp_min(1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--F", type=int, default=100)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--d-embed", type=int, default=50)
    ap.add_argument("--P", type=float, default=0.02)
    ap.add_argument("--loss-exp", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-seed", type=int, default=12345)
    args = ap.parse_args()
    print(f"device: {DEVICE}")

    W_E = random_embedding(args.F, args.d_embed, args.embed_seed)
    G = (W_E @ W_E.T).cpu().numpy()
    off = G[~np.eye(args.F, dtype=bool)]
    print(f"W_E gram off-diag: mean|.|={np.abs(off).mean():.4f} "
          f"max|.|={np.abs(off).max():.4f}  (1/sqrt(d)={1/np.sqrt(args.d_embed):.4f})")

    Win, Wout = train(args.F, args.N, args.P, args.d_embed,
                      args.loss_exp, args.steps, W_E, seed=args.seed)

    mse = per_feature_mse(Win, Wout, W_E, args.F, args.P)
    cv = float(mse.std() / mse.mean())
    # naive baseline: learn floor(N) features (or as many as fit), ignore rest -> 1/3 on ignored
    print(f"\nper-feature MSE: mean={mse.mean():.4e}  max={mse.max():.4e}  "
          f"min={mse.min():.4e}  CV={cv:.3f}")
    print(f"CiS verdict: {'YES (low CV, error spread evenly)' if cv < 0.2 else 'NO (high CV, naive-like)'}")

    tag = f"F{args.F}_N{args.N}_d{args.d_embed}_L{args.loss_exp}"
    out = Path(f"data/dim_{tag}.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez(out, mse=mse, meta=json.dumps(dict(
        F=args.F, N=args.N, d_embed=args.d_embed, P=args.P,
        loss_exp=args.loss_exp, steps=args.steps, cv=cv)))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
