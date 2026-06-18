"""Train many (loss-exponent, seed) toy models and dump per-feature MSE.

For each loss exponent we train ``--seeds`` independent nets, vectorized over
the seed axis (one big batched matmul), then evaluate per-feature MSE on a
shared fixed eval set. Results are written to a single .npz:

    perfeat[str(exp)]  ->  float array [n_seeds, F]   per-feature MSE on x>0
    meta               ->  json blob with the training recipe

Seeds vary BOTH the weight init and the training data: each net trains on its
own fresh per-step data stream (its slice of the batched data tensor). Across
exponents the master seed is reset, so seed s sees the same init + data stream
for every exponent (a paired design that isolates the effect of the exponent).

This matches train_from_scratch.py's recipe (Adam lr 0.01, cosine annealing,
batch 8192, 100k steps, F=100 N=50 p=0.02, same init scales) except that data
is streamed fresh each step rather than sampled from a fixed 800k pool.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from small_models import random_embedding

F, N, P = 100, 50, 0.02
STEPS = 100_000
LR = 0.01
BATCH = 8192
MASTER_SEED = 0
D_EMBED = 1000           # Braun et al. embedding dimension (0 = axis-aligned)
EMBED_SEED = 12345       # fixed embedding shared across all seeds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# eval set (shared across all nets for a clean comparison)
EVAL_SEED = 9999
EVAL_BATCHES = 100
EVAL_BATCH = 2048


def gen_batch(n_seeds, batch, device):
    """Sparse Bernoulli-Uniform batch of shape [n_seeds, batch, F]."""
    mask = (torch.rand(n_seeds, batch, F, device=device) < P).float()
    vals = torch.rand(n_seeds, batch, F, device=device) * 2 - 1
    x = mask * vals
    return x, torch.relu(x)


def make_eval_set(device):
    """Fixed eval batches shared by every trained net (no seed axis)."""
    g = torch.Generator(device=device).manual_seed(EVAL_SEED)
    batches = []
    for _ in range(EVAL_BATCHES):
        mask = (torch.rand(EVAL_BATCH, F, device=device, generator=g) < P).float()
        vals = torch.rand(EVAL_BATCH, F, device=device, generator=g) * 2 - 1
        x = mask * vals
        batches.append((x, torch.relu(x)))
    return batches


def train_seeds(loss_exp, n_seeds, steps, device, W_E=None):
    """Train n_seeds nets for one loss exponent, vectorized over seeds.

    With W_E (F, d) the model reads/writes a d-dim residual stream:
        y = relu((x @ W_E) @ W_in.T) @ W_out.T @ W_E.T
    W_in/W_out then have inner dim d instead of F. Returns (W_in, W_out) with
    shapes [S, N, d] / [S, d, N]  (d = F when W_E is None).
    """
    torch.manual_seed(MASTER_SEED)
    S = n_seeds
    d = F if W_E is None else W_E.shape[1]
    W_in = (torch.rand(S, N, d, device=device) * 0.2 - 0.1).requires_grad_()
    W_out = (torch.rand(S, d, N, device=device) * 0.3 - 0.15).requires_grad_()
    opt = torch.optim.Adam([W_in, W_out], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    log_every = max(1, steps // 5)
    for step in range(steps):
        x, y = gen_batch(S, BATCH, device)
        r = x if W_E is None else x @ W_E               # [S,B,F] -> [S,B,d]
        h = torch.relu(torch.einsum("snd,sbd->sbn", W_in, r))
        out_r = torch.einsum("sdn,sbn->sbd", W_out, h)
        out = out_r if W_E is None else out_r @ W_E.T    # [S,B,d] -> [S,B,F]
        loss = ((out - y).abs() ** loss_exp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            print(f"  exp={loss_exp:<4} step {step + 1:>6}/{steps}  "
                  f"L{loss_exp} = {loss.item():.4e}", flush=True)
    return W_in.detach(), W_out.detach()


@torch.no_grad()
def per_feature_mse(W_in, W_out, eval_set, W_E=None):
    """Per-feature MSE conditioned on the feature being active, per seed.

    W_in:[S,N,d]  W_out:[S,d,N]  ->  [S, F].
    """
    S = W_in.shape[0]
    err = torch.zeros(S, F, device=W_in.device)
    cnt = torch.zeros(F, device=W_in.device)
    for x, y in eval_set:                       # x,y: [B, F] shared
        r = x if W_E is None else x @ W_E       # [B,F] -> [B,d]
        h = torch.relu(torch.einsum("snd,bd->sbn", W_in, r))
        out_r = torch.einsum("sdn,sbn->sbd", W_out, h)
        out = out_r if W_E is None else out_r @ W_E.T
        active = (x > 0).float()                # [B, F]
        err += ((out - y) ** 2 * active.unsqueeze(0)).sum(dim=1)
        cnt += active.sum(dim=0)
    return (err / cnt.clamp_min(1).unsqueeze(0)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exponents", type=float, nargs="+",
                    default=[1, 2, 2.5, 3, 4, 6, 8])
    ap.add_argument("--seeds", type=int, default=10,
                    help="nets per exponent (fig1 uses 10; fig2 uses first 5)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--d-embed", type=int, default=D_EMBED,
                    help="residual-stream dim for the fixed Braun embedding; "
                         "0 = axis-aligned (original setup)")
    ap.add_argument("--embed-seed", type=int, default=EMBED_SEED)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    W_E = None
    if args.d_embed > 0:
        W_E = random_embedding(F, args.d_embed, args.embed_seed, device=DEVICE)
        print(f"fixed embedding W_E {tuple(W_E.shape)} (seed {args.embed_seed}), "
              f"shared across all seeds")
    out_path = args.out or ("data/seed_experiments_embedded.npz" if W_E is not None
                            else "data/seed_experiments.npz")
    eval_set = make_eval_set(DEVICE)

    perfeat = {}
    for exp in args.exponents:
        print(f"=== loss exponent {exp} ({args.seeds} seeds) ===", flush=True)
        W_in, W_out = train_seeds(exp, args.seeds, args.steps, DEVICE, W_E=W_E)
        mse = per_feature_mse(W_in, W_out, eval_set, W_E=W_E)        # [S, F]
        cv = mse.std(axis=1, ddof=0) / mse.mean(axis=1)
        print(f"  per-feature MSE mean={mse.mean():.4e}  "
              f"CV per seed: mean={cv.mean():.3f} std={cv.std(ddof=1):.3f}",
              flush=True)
        perfeat[str(exp)] = mse

    meta = dict(F=F, N=N, P=P, steps=args.steps, lr=LR, batch=BATCH,
                master_seed=MASTER_SEED, seeds=args.seeds,
                exponents=args.exponents, eval_seed=EVAL_SEED,
                eval_batches=EVAL_BATCHES, eval_batch=EVAL_BATCH,
                d_embed=(0 if W_E is None else args.d_embed),
                embed_seed=args.embed_seed,
                note="data streamed fresh per step; seeds vary init+data; "
                     "eval set shared across all nets; fixed Braun embedding "
                     "shared across seeds when d_embed>0")
    out = Path(out_path)
    out.parent.mkdir(exist_ok=True)
    np.savez(out, meta=json.dumps(meta),
             **{f"exp_{k}": v for k, v in perfeat.items()})
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
