"""Train the no-embedding model with W_out TIED to a scaled pseudoinverse of W_in.

Stefan's TODO from the feedback: instead of letting the decoder train freely,
tie it to W_out = c * pinv(W_in) and recompute the pseudoinverse every step
(cheap in the no-W_E case, where W_in is 50x100). Only W_in and the scalar c
are free parameters. This isolates how much loss the freely-trained decoder
buys over an exact (scaled) pseudoinverse decoder.

  python train_tied_pinv.py --loss-exp 4

Mirrors train_from_scratch.py exactly in data, optimizer, schedule, and seeds,
so the resulting L4 loss is directly comparable to
weights/codeword_100f_50n_L4_100000steps.pt (the freely-trained no-embed model).
Prints the final L4 loss and the ratio to that trained model.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from small_models import generate_batch, DEVICE

F, N, P, TAU = 100, 50, 0.02, 0.05
STEPS = 100_000
LR = 0.01
BATCH = 8192


def l4_on(W_in, W_out, x, y, exp):
    return ((torch.relu(x @ W_in.T) @ W_out.T - y).abs() ** exp).mean()


def train(loss_exp, steps=STEPS, seed=0, data_seed=31337):
    torch.manual_seed(data_seed)
    x_tr, y_tr = generate_batch(800_000, F, P)
    x_ev, y_ev = generate_batch(600_000, F, P)

    torch.manual_seed(seed)
    W_in = (torch.rand(N, F, device=DEVICE) * 0.2 - 0.1).requires_grad_()
    # decoder scale, initialised near the trained pinv scale (~1.6)
    c = torch.tensor(1.6, device=DEVICE).requires_grad_()
    opt = torch.optim.Adam([W_in, c], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    log_every = max(1, steps // 10)
    print(f"training tied-pinv L{loss_exp} model, {steps} steps...", flush=True)
    for step in range(steps):
        idx = torch.randint(0, x_tr.shape[0], (BATCH,), device=DEVICE)
        W_out = c * torch.linalg.pinv(W_in)            # (F, N), recomputed each step
        loss = l4_on(W_in, W_out, x_tr[idx], y_tr[idx], loss_exp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            with torch.no_grad():
                W_out = c * torch.linalg.pinv(W_in)
                ev = l4_on(W_in, W_out, x_ev, y_ev, loss_exp).item()
                cw = (W_in.detach().cpu().numpy().T > TAU)
                csize = cw.sum(1)
                kd = " ".join(f"K{k}:{(csize == k).sum()}"
                              for k in sorted(set(csize.tolist())))
            print(f"  step {step + 1:>6}/{steps}  L{loss_exp} = {ev:.4e}  "
                  f"c = {c.item():.3f}  K-dist={kd}", flush=True)

    with torch.no_grad():
        W_out = c * torch.linalg.pinv(W_in)
    return W_in.detach(), W_out.detach(), c.detach(), (x_ev, y_ev)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loss-exp", type=int, default=4)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=31337)
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    np.random.seed(args.seed)

    W_in, W_out, c, (x_ev, y_ev) = train(
        args.loss_exp, steps=args.steps, seed=args.seed, data_seed=args.data_seed)

    tied_l4 = l4_on(W_in, W_out, x_ev, y_ev, args.loss_exp).item()

    # Compare against the freely-trained no-embed model on the same eval data.
    ref_path = Path("weights/codeword_100f_50n_L4_100000steps.pt")
    ratio_str = ""
    if ref_path.exists():
        ref = torch.load(ref_path)
        ref_l4 = l4_on(ref["W_in"].to(DEVICE), ref["W_out"].to(DEVICE),
                       x_ev, y_ev, args.loss_exp).item()
        ratio_str = f"  | trained(free)={ref_l4:.4e}  ratio={tied_l4 / ref_l4:.3f}x"

    print(f"\nFINAL tied-pinv L{args.loss_exp} = {tied_l4:.4e}{ratio_str}")

    out_path = Path(f"weights/tied_pinv_100f_50n_L{args.loss_exp}_{args.steps}steps.pt")
    out_path.parent.mkdir(exist_ok=True)
    torch.save({"W_in": W_in.cpu(), "W_out": W_out.cpu(), "c": c.cpu()}, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
