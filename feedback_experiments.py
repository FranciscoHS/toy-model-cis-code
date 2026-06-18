"""Paper-feedback experiments on the embedded CiS model (items 1, 4, 5).

Everything runs on the embedded checkpoint's effective weights W_in_eff (N,F),
W_out_eff (F,N) -- the codeword matrix -- which are an exact axis-aligned
equivalent of the embedded model. Losses are reported in the paper's own
conventions: total mean L4 over all samples, and per-feature MSE on x>0 samples.

  item 1  emulate-bias baseline (Bhagat et al. "naive + offset") + do-nothing /
          random references; saves the per-feature MSE for the Fig 2 overlay.
  item 4  replace decoder with a scaled pseudoinverse of the encoder.
  item 5  fix the codeword support M, retrain individual encoder values
          (decoder = scaled pinv), measure loss vs trained (1x) and 3-scalar.

    python feedback_experiments.py weights/embedded_100f_50n_d1000_L4_100000steps.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from small_models import generate_batch, evaluate_per_feature_mse, DEVICE

F, N, P, TAU = 100, 50, 0.02, 0.05


def eval_l4(W_in, W_out, x, y):
    with torch.no_grad():
        yhat = torch.relu(x @ W_in.T) @ W_out.T
        return ((yhat - y).abs() ** 4).mean().item()


def pf(W_in, W_out):
    """Per-feature MSE on x>0 (numpy [F])."""
    return evaluate_per_feature_mse(W_in, W_out, F, P)


# ----------------------------------------------------------------------------- item 1
def emulate_bias_baseline(x_tr, y_tr, steps=4000):
    """Bhagat 'naive + offset': 50 features represented by identity, the other 50
    output a constant offset emulated by constant W_out rows over all neurons.
    Free scalars: a (represented decode scale), c (offset strength)."""
    rep = torch.arange(N)                                  # features 0..49 represented
    W_in = torch.zeros(N, F, device=DEVICE)
    W_in[torch.arange(N), rep] = 1.0                       # identity encoder on rep set
    rep_mask = torch.zeros(F, N, device=DEVICE)
    rep_mask[rep, torch.arange(N)] = 1.0                   # represented rows: own neuron
    off_mask = torch.ones(F, N, device=DEVICE)
    off_mask[rep] = 0.0                                    # unrepresented rows: all neurons

    a = torch.tensor(1.0, device=DEVICE, requires_grad=True)
    c = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([a, c], lr=0.01)
    for _ in range(steps):
        idx = torch.randint(0, x_tr.shape[0], (8192,), device=DEVICE)
        W_out = a * rep_mask + c * off_mask
        yhat = torch.relu(x_tr[idx] @ W_in.T) @ W_out.T
        loss = ((yhat - y_tr[idx]).abs() ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        W_out = a * rep_mask + c * off_mask
    return W_in, W_out.detach(), (a.item(), c.item())


def random_baseline(seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.rand(N, F, generator=g, device=DEVICE) * 0.2 - 0.1
    W_out = torch.rand(F, N, generator=g, device=DEVICE) * 0.3 - 0.15
    return W_in, W_out


# ----------------------------------------------------------------------------- item 4
def pinv_decoder(W_in_eff, x_tr, y_tr, steps=3000):
    """Decoder := s * pinv(encoder); fit scalar s for L4."""
    pinv = torch.linalg.pinv(W_in_eff)                     # (F, N)
    s = torch.tensor(1.0, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([s], lr=0.02)
    for _ in range(steps):
        idx = torch.randint(0, x_tr.shape[0], (8192,), device=DEVICE)
        yhat = torch.relu(x_tr[idx] @ W_in_eff.T) @ (s * pinv).T
        loss = ((yhat - y_tr[idx]).abs() ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W_in_eff, (s.detach() * pinv), s.item()


# ----------------------------------------------------------------------------- item 5
def retrain_values_fixed_M(W_in_eff, x_tr, y_tr, steps=15000, lam=1.0):
    """Fix the codeword support M; free per-entry encoder values (init = trained),
    decoder = s * pinv(encoder). A margin penalty keeps on-code entries positive
    and off-code entries small so M does not flip."""
    M = (W_in_eff > TAU)                                   # (N, F) support
    on, off = M, ~M
    W = W_in_eff.clone().requires_grad_()
    s = torch.tensor(1.0, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([W, s], lr=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, x_tr.shape[0], (8192,), device=DEVICE)
        dec = s * torch.linalg.pinv(W)
        yhat = torch.relu(x_tr[idx] @ W.T) @ dec.T
        l4 = ((yhat - y_tr[idx]).abs() ** 4).mean()
        # margin: on-code >= TAU+margin, off-code <= TAU-margin
        pen = (torch.relu((TAU + 0.02) - W[on]).pow(2).sum()
               + torch.relu(W[off] - (TAU - 0.02)).pow(2).sum())
        (l4 + lam * pen / W.numel()).backward()
        opt.step(); opt.zero_grad(); sched.step()
    with torch.no_grad():
        dec = s * torch.linalg.pinv(W)
        M_after = (W > TAU)
        flipped = int((M_after != M).sum())
    return W.detach(), dec.detach(), flipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=str)
    ap.add_argument("--out", type=str, default="data/feedback_experiments.npz")
    args = ap.parse_args()
    print(f"device: {DEVICE}")

    sd = torch.load(args.weights, map_location=DEVICE)
    W_in_eff = sd["W_in_eff"].to(DEVICE)
    W_out_eff = sd["W_out_eff"].to(DEVICE)

    torch.manual_seed(31337)
    x_tr, y_tr = generate_batch(600_000, F, P)
    x_ev, y_ev = generate_batch(600_000, F, P)

    l4_trained = eval_l4(W_in_eff, W_out_eff, x_ev, y_ev)
    print(f"\nembedded trained L4 = {l4_trained:.4e}  (= 1.00x reference)")

    results = {"l4_trained": l4_trained}

    # ---- item 1: baselines ----
    print("\n=== item 1: baselines ===")
    Wi0 = torch.zeros(N, F, device=DEVICE)
    l4_donothing = eval_l4(Wi0, torch.zeros(F, N, device=DEVICE), x_ev, y_ev)
    pf_donothing = pf(Wi0, torch.zeros(F, N, device=DEVICE))
    rnd = [eval_l4(*random_baseline(s), x_ev, y_ev) for s in range(5)]
    l4_random = float(np.mean(rnd))
    Wi_b, Wo_b, (a, c) = emulate_bias_baseline(x_tr, y_tr)
    l4_bias = eval_l4(Wi_b, Wo_b, x_ev, y_ev)
    pf_bias = pf(Wi_b, Wo_b)
    # naive (ignore half) = emulate-bias with no offset (a=1, c=0): represent 50,
    # zero the other 50. This is the paper's "naive ~28x" baseline.
    rep = torch.arange(N)
    Wi_naive = torch.zeros(N, F, device=DEVICE); Wi_naive[torch.arange(N), rep] = 1.0
    Wo_naive = torch.zeros(F, N, device=DEVICE); Wo_naive[rep, torch.arange(N)] = 1.0
    l4_naive = eval_l4(Wi_naive, Wo_naive, x_ev, y_ev)
    pf_naive = pf(Wi_naive, Wo_naive)
    for name, l4 in [("do-nothing", l4_donothing), ("random", l4_random),
                     ("naive(ignore½)", l4_naive), ("emulate-bias", l4_bias)]:
        print(f"  {name:<14} L4={l4:.4e}  = {l4 / l4_trained:6.1f}x trained")
    results.update(l4_naive=l4_naive, pf_naive=pf_naive)
    print(f"  emulate-bias (a,c)=({a:.3f},{c:.4f}); per-feat MSE mean={pf_bias.mean():.4f} "
          f"max={pf_bias.max():.4f}")
    results.update(l4_donothing=l4_donothing, l4_random=l4_random, l4_bias=l4_bias,
                   pf_bias=pf_bias, pf_donothing=pf_donothing, bias_ac=[a, c])

    # ---- item 4: pinv decoder ----
    print("\n=== item 4: pseudoinverse decoder ===")
    Wi_p, Wo_p, s = pinv_decoder(W_in_eff, x_tr, y_tr)
    l4_pinv = eval_l4(Wi_p, Wo_p, x_ev, y_ev)
    print(f"  decoder = {s:.3f} * pinv(encoder):  L4={l4_pinv:.4e}  "
          f"= {l4_pinv / l4_trained:.3f}x trained")
    results.update(l4_pinv=l4_pinv, pinv_s=s)

    # ---- item 5: fixed M, retrained values ----
    print("\n=== item 5: fix M, retrain per-entry values (decoder = scaled pinv) ===")
    Wi_r, Wo_r, flipped = retrain_values_fixed_M(W_in_eff, x_tr, y_tr)
    l4_retrain = eval_l4(Wi_r, Wo_r, x_ev, y_ev)
    print(f"  retrained values:  L4={l4_retrain:.4e}  = {l4_retrain / l4_trained:.3f}x "
          f"trained   (M flipped entries: {flipped})")
    results.update(l4_retrain=l4_retrain, M_flipped=flipped)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    np.savez(out, **{k: v for k, v in results.items()
                     if isinstance(v, np.ndarray)},
             meta=json.dumps({k: v for k, v in results.items()
                              if not isinstance(v, np.ndarray)}))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
