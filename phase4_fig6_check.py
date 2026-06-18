"""Fig 6 normalization check for the embedded model.

Fig 6 (synthetic codes) normalizes ansatz losses by the trained model's L4 loss.
The synthetic-code ansatz lives in (N,F) space and is unchanged by the embedding
(lossless basis change), so the sweep itself need not be re-run; only the trained
baseline could move. This script:

  1. recomputes the trained L4 baseline from the embedded effective weights on the
     same eval distribution sweep_K_codes.py uses, and compares to the axis-aligned
     baseline stored in data/sweep_K_codes.json;
  2. extracts the trained binary code M from the embedded effective encoder and fits
     the 3-parameter ansatz to it (the paper's "~1.13x trained loss" claim).
"""
import json
import numpy as np
import torch

from small_models import generate_batch, DEVICE
from sweep_K_codes import train_3param

F, N, P = 100, 50, 0.02
TAU = 0.05
CKPT = "weights/embedded_100f_50n_d1000_L4_100000steps.pt"


def main():
    print(f"device: {DEVICE}")
    sd = torch.load(CKPT, map_location=DEVICE)
    W_in_eff = sd["W_in_eff"].to(DEVICE)        # (N, F)
    W_out_eff = sd["W_out_eff"].to(DEVICE)      # (F, N)

    torch.manual_seed(31337)
    x_tr, y_tr = generate_batch(600_000, F, P)
    x_ev, y_ev = generate_batch(600_000, F, P)

    with torch.no_grad():
        l4_emb = ((torch.relu(x_ev @ W_in_eff.T) @ W_out_eff.T - y_ev)
                  .abs() ** 4).mean().item()

    axis = json.load(open("data/sweep_K_codes.json"))["meta"]["l4_trained_100k"]
    print(f"\ntrained L4 baseline:")
    print(f"  axis-aligned (in JSON) = {axis:.4e}")
    print(f"  embedded (effective)   = {l4_emb:.4e}")
    print(f"  ratio embedded/axis    = {l4_emb / axis:.4f}  "
          f"(Fig 6 y-values scale by 1/this)")

    # trained-code 3-param ansatz: M from the embedded effective encoder
    M = (W_in_eff.cpu().numpy().T > TAU)         # (F, N) boolean codeword matrix
    K = M.sum(1)
    print(f"\ntrained code M from embedded encoder: "
          f"K range {K.min()}-{K.max()}, mean {K.mean():.2f}")
    l4_ansatz, abc = train_3param(M, x_tr, y_tr, x_ev, y_ev)
    print(f"3-param ansatz L4 = {l4_ansatz:.4e}  "
          f"= {l4_ansatz / l4_emb:.3f}x embedded trained  (paper: ~1.13x)")
    print(f"  (a,b,c) = {tuple(round(v, 4) for v in abc)}")


if __name__ == "__main__":
    main()
