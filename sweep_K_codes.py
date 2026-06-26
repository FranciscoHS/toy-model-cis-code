"""Sweep L4 loss vs codeword length K for two code families:

  - regular (birregular bipartite, balanced feature- and neuron-degree)
  - random  (each feature picks K random neurons, no neuron-degree constraint)

For each (code, K, seed) we fit the same 3-parameter model used elsewhere:

    W_in  = a * M.T + b * (1 - M.T)
    W_out = c * pinv(M.T)

and report total mean L4 loss on a held-out batch. Results are written to
``data/sweep_K_codes.json`` so the plotter can run separately.
"""
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from small_models import generate_batch, DEVICE
from codes import regular_code, random_code, reduce_overlap

F, N, P = 100, 50, 0.02
K_LIST = [4, 5, 6, 7]
SEEDS = list(range(5))
STEPS = 15_000
SWAPS = 800_000          # reduce_overlap iterations, applied to BOTH families
EVAL_SAMPLES = 600_000
TRAIN_SAMPLES = 600_000
OUT = Path("data/sweep_K_codes.json")


def train_3param(M, x_tr, y_tr, x_ev, y_ev, steps=STEPS):
    """Fit (a, b, c) for the W_in = a M.T + b (1-M.T), W_out = c * pinv(M.T)
    ansatz against L4 loss with Adam + cosine LR."""
    M_NF = torch.tensor(M.T.astype(np.float32), device=DEVICE)
    M_f = M.astype(np.float64)
    PINV = np.linalg.pinv(M_f.T).astype(np.float32)         # (F, N)
    PINV_t = torch.tensor(PINV, device=DEVICE)
    a = torch.tensor(0.85, device=DEVICE, requires_grad=True)
    b = torch.tensor(-0.04, device=DEVICE, requires_grad=True)
    c = torch.tensor(1.9, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([a, b, c], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, x_tr.shape[0], (8192,), device=DEVICE)
        Wi = a * M_NF + b * (1 - M_NF)
        Wo = c * PINV_t
        loss = ((torch.relu(x_tr[idx] @ Wi.T) @ Wo.T - y_tr[idx]).abs() ** 4).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        Wi = a * M_NF + b * (1 - M_NF)
        Wo = c * PINV_t
        l4 = ((torch.relu(x_ev @ Wi.T) @ Wo.T - y_ev).abs() ** 4).mean().item()
    return l4, (a.item(), b.item(), c.item())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--swaps", type=int, default=SWAPS,
                    help="reduce_overlap iterations (0 = no edge-swap / no overlap minimization)")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    swaps = args.swaps
    out_path = Path(args.out)

    print(f"device: {DEVICE}  swaps={swaps}  out={out_path}")
    out_path.parent.mkdir(exist_ok=True)
    torch.manual_seed(31337)
    x_tr, y_tr = generate_batch(TRAIN_SAMPLES, F, P)
    x_ev, y_ev = generate_batch(EVAL_SAMPLES, F, P)

    sd = torch.load("weights/codeword_100f_50n_L4_100000steps.pt",
                    map_location=DEVICE)
    Wi_t = sd["W_in"].to(DEVICE)
    Wo_t = sd["W_out"].to(DEVICE)
    with torch.no_grad():
        l4_trained = ((torch.relu(x_ev @ Wi_t.T) @ Wo_t.T - y_ev).abs() ** 4).mean().item()
    print(f"trained 100k baseline L4 = {l4_trained:.4e}")

    results = {
        "meta": dict(
            F=F, N=N, P=P, K_list=K_LIST, seeds=SEEDS,
            steps=STEPS, swaps=swaps,
            train_samples=TRAIN_SAMPLES, eval_samples=EVAL_SAMPLES,
            l4_trained_100k=l4_trained,
        ),
        "regular": {},
        "random": {},
    }

    t0 = time.time()
    for K in K_LIST:
        for code_type in ("regular", "random"):
            losses, params, sqs, maxovs = [], [], [], []
            for seed in SEEDS:
                if code_type == "regular":
                    base = regular_code(F, N, K, seed)
                else:
                    base = random_code(F, N, K, seed)
                M = base if swaps == 0 else reduce_overlap(base, swaps, seed=10 + seed)
                ov = M.astype(int) @ M.astype(int).T
                np.fill_diagonal(ov, 0)
                sq = int((ov ** 2).sum())
                mxov = int(ov.max())
                l4, abc = train_3param(M, x_tr, y_tr, x_ev, y_ev)
                losses.append(l4)
                params.append(abc)
                sqs.append(sq)
                maxovs.append(mxov)
                elapsed = time.time() - t0
                print(f"  K={K} {code_type:>7} seed={seed}  L4={l4:.4e}  "
                      f"frac={l4 / l4_trained:.3f}x  sum(ov^2)={sq}  maxOv={mxov}  "
                      f"[t={elapsed:.0f}s]", flush=True)
            results[code_type][str(K)] = dict(
                losses=losses, params=params, sq=sqs, maxov=maxovs,
            )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}  (elapsed {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
