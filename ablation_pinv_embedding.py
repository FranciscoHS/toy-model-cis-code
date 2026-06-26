"""Ablation: RETRAIN the embedded model with a pseudoinverse unembedding instead
of the transpose, holding the optimizer, schedule, budget, embedding and seed
fixed -- so the only change is the read-out matrix.

Embedded forward (transpose, the paper's model):
    r = x @ W_E ;  h = relu(r @ W_in.T) @ W_out.T ;  y = h @ W_E.T
Ablation (pinv): replace the read-out W_E.T with pinv(W_E) and retrain W_in,W_out.

pinv(W_E) is the exact left-inverse of the embedding (removes the small residual
cross-talk between the near-orthogonal feature directions that the transpose
leaves in). The question is whether, after retraining, the network reaches the
same loss/solution -- i.e. whether the transpose-vs-pinv read-out is incidental.

Reports final L4 loss, per-feature MSE (mean/max) and its coefficient of
variation for both read-outs, and the pinv/transpose loss ratio.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from small_models import generate_batch, random_embedding, DEVICE

F, N, P, D_EMBED, LR, BATCH = 100, 50, 0.02, 1000, 0.01, 8192


def train(unembed, loss_exp, steps, W_E, seed=0, data_seed=31337):
    torch.manual_seed(data_seed)
    x_tr, y_tr = generate_batch(800_000, F, P)
    x_ev, y_ev = generate_batch(200_000, F, P)
    torch.manual_seed(seed)
    W_in = (torch.rand(N, D_EMBED, device=DEVICE) * 0.2 - 0.1).requires_grad_()
    W_out = (torch.rand(D_EMBED, N, device=DEVICE) * 0.3 - 0.15).requires_grad_()
    opt = torch.optim.Adam([W_in, W_out], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def forward(x):
        h = torch.relu((x @ W_E) @ W_in.T) @ W_out.T      # (B, d)
        return h @ unembed                                 # (B, F)

    log_every = max(1, steps // 10)
    for step in range(steps):
        idx = torch.randint(0, x_tr.shape[0], (BATCH,), device=DEVICE)
        loss = ((forward(x_tr[idx]) - y_tr[idx]).abs() ** loss_exp).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            with torch.no_grad():
                ev = ((forward(x_ev) - y_ev).abs() ** loss_exp).mean().item()
            print(f"  step {step + 1:>6}/{steps}  L{loss_exp} = {ev:.4e}", flush=True)

    with torch.no_grad():
        l4 = ((forward(x_ev) - y_ev).abs() ** 4).mean().item()
        torch.manual_seed(9999)
        err = torch.zeros(F, device=DEVICE); cnt = torch.zeros(F, device=DEVICE)
        for _ in range(100):
            x, y = generate_batch(2048, F, P)
            yhat = forward(x); active = (x > 0).float()
            err += ((yhat - y) ** 2 * active).sum(0); cnt += active.sum(0)
        pf = (err / cnt.clamp_min(1)).cpu().numpy()
    return dict(l4=l4, pf_mean=float(pf.mean()), pf_max=float(pf.max()),
                cv=float(pf.std() / pf.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--embed-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("device:", DEVICE, flush=True)

    W_E = random_embedding(F, D_EMBED, a.embed_seed)
    pinvE = torch.linalg.pinv(W_E)
    G = (W_E @ W_E.T)
    print(f"W_E gram off-diag max|.|={ (G - torch.eye(F, device=DEVICE)).abs().max().item():.4f}", flush=True)

    res = {}
    for name, unembed in [("transpose", W_E.T), ("pinv", pinvE)]:
        print(f"=== training {name} unembedding (L4, {a.steps} steps) ===", flush=True)
        res[name] = train(unembed, 4, a.steps, W_E, seed=a.seed)
        print(name, res[name], flush=True)
    res["ratio_l4_pinv_over_transpose"] = res["pinv"]["l4"] / res["transpose"]["l4"]

    Path("results").mkdir(exist_ok=True)
    out = Path("results/ablation_pinv_embedding.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n" + json.dumps(res, indent=2), flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
