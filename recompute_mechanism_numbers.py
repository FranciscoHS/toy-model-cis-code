"""Recompute the trained-model mechanism numbers for the embedded model.

Loads an embedded checkpoint, collapses it to effective (N,F)/(F,N) weights, and
reports the quantities the paper asserts (Figs 4-5 and surrounding text):

  - swap test pass rate (Fig 4): argmax(decode(transplant)) == target
  - decoder vs scaled-pinv-of-encoder Frobenius cosine (paper: 0.996)
  - on/off-codeword encoder values (paper: ~0.3-0.4 / ~-0.03)

    python recompute_mechanism_numbers.py weights/embedded_100f_50n_d1000_L4_100000steps.pt
"""
import argparse
import numpy as np
import torch

from swap_test import swap_test, print_report


def frobenius_cosine(A, B):
    a, b = A.flatten(), B.flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=str)
    args = ap.parse_args()

    sd = torch.load(args.weights, map_location="cpu")
    if "W_in_eff" in sd:
        W_in = sd["W_in_eff"].numpy()        # (N, F)
        W_out = sd["W_out_eff"].numpy()      # (F, N)
        print(f"embedded checkpoint: using effective weights "
              f"W_in_eff{W_in.shape}, W_out_eff{W_out.shape}")
    else:
        W_in = sd["W_in"].numpy()
        W_out = sd["W_out"].numpy()
        print(f"axis-aligned checkpoint: W_in{W_in.shape}, W_out{W_out.shape}")

    # --- Fig 4: swap test ---
    print("\n=== swap test (Fig 4) ===")
    print_report(swap_test(W_in, W_out))

    # --- decoder vs scaled pinv of encoder (mechanism text) ---
    pinv = np.linalg.pinv(W_in)              # (N, F).T -> pinv of (N,F) is (F,N)
    cos = frobenius_cosine(W_out, pinv)
    print(f"\n=== decoder vs pinv(encoder) ===")
    print(f"Frobenius cosine(W_out, pinv(W_in)) = {cos:.4f}  (paper: 0.996)")

    # --- on/off-codeword encoder values (Fig 3 / mechanism) ---
    TAU = 0.05
    on = W_in[W_in > TAU]
    off = W_in[W_in <= TAU]
    csize = (W_in > TAU).sum(axis=0)
    kd = " ".join(f"K{k}:{(csize == k).sum()}" for k in sorted(set(csize.tolist())))
    print(f"\n=== encoder codeword stats (tau={TAU}) ===")
    print(f"K-dist: {kd}")
    print(f"on-code  mean={on.mean():+.4f}  off-code mean={off.mean():+.4f}")


if __name__ == "__main__":
    main()
