"""Train the F=100, N=50, p=0.02 single-hidden-layer model from scratch.

  python train_from_scratch.py --loss-exp 4
  python train_from_scratch.py --loss-exp 2

Produces ``weights/codeword_100f_50n_L{loss_exp}_{STEPS}steps.pt`` containing
``{"W_in": (N, F), "W_out": (F, N)}``. These checkpoints are what the plot
scripts load. The L4 run also prints, at each logged step, the codeword-size
distribution after thresholding ``W_in`` at ``TAU``, which is a useful
sanity check that codewords settle to a narrow K range.
"""
import argparse
import torch
import numpy as np
from pathlib import Path

from small_models import generate_batch, DEVICE

F, N, P, TAU = 100, 50, 0.02, 0.05
STEPS = 100_000
LR = 0.01
BATCH = 8192


def train(loss_exp, steps=STEPS, seed=0, data_seed=31337, log_K_dist=True):
    """Train and return (W_in, W_out) as detached tensors on DEVICE."""
    torch.manual_seed(data_seed)
    x_tr, y_tr = generate_batch(800_000, F, P)
    x_ev, y_ev = generate_batch(600_000, F, P)

    torch.manual_seed(seed)
    W_in = (torch.rand(N, F, device=DEVICE) * 0.2 - 0.1).requires_grad_()
    W_out = (torch.rand(F, N, device=DEVICE) * 0.3 - 0.15).requires_grad_()
    opt = torch.optim.Adam([W_in, W_out], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    log_every = max(1, steps // 10)
    print(f"training L{loss_exp} model, {steps} steps...")
    for step in range(steps):
        idx = torch.randint(0, x_tr.shape[0], (BATCH,), device=DEVICE)
        loss = ((torch.relu(x_tr[idx] @ W_in.T) @ W_out.T - y_tr[idx]).abs() ** loss_exp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            with torch.no_grad():
                ev = ((torch.relu(x_ev @ W_in.T) @ W_out.T - y_ev).abs() ** loss_exp).mean().item()
            msg = f"  step {step + 1:>6}/{steps}  L{loss_exp} = {ev:.4e}"
            if log_K_dist:
                cw = (W_in.detach().cpu().numpy().T > TAU)
                csize = cw.sum(1)
                kd = " ".join(f"K{k}:{(csize == k).sum()}" for k in sorted(set(csize.tolist())))
                msg += f"  K-dist={kd}"
            print(msg, flush=True)

    return W_in.detach(), W_out.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loss-exp", type=int, choices=[2, 4], required=True,
                    help="loss exponent (2 = MSE, 4 = quartic)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=31337,
                    help="seed for the train/eval data batches")
    ap.add_argument("--out", type=str, default=None,
                    help="output .pt path (default: "
                         "weights/codeword_100f_50n_L<loss-exp>_<steps>steps.pt)")
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    np.random.seed(args.seed)

    W_in, W_out = train(args.loss_exp, steps=args.steps,
                        seed=args.seed, data_seed=args.data_seed)

    out_path = Path(args.out) if args.out else \
        Path(f"weights/codeword_100f_50n_L{args.loss_exp}_{args.steps}steps.pt")
    out_path.parent.mkdir(exist_ok=True)
    torch.save({"W_in": W_in.cpu(), "W_out": W_out.cpu()}, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
