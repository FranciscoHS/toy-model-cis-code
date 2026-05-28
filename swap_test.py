"""Codeword-pattern swap test.

For pairs of features (i, j) with the same number of active neurons K:

  1. Compute h_i = relu(W_in @ e_i): the hidden activation produced when
     only feature i is on.
  2. Take the K nonzero values of h_i (ordered by neuron index ascending).
  3. Place those K values at j's K active neuron positions (also ordered by
     neuron index ascending). All other positions are zero. Call this h_swap.
  4. y = W_out @ h_swap.
  5. Ask: does argmax(y) equal j (the *positions* won) or i (the *values*
     won) or some third feature?

A model that uses the codeword positions as the read-out cue, with the
on-codeword values being roughly interchangeable, should produce
argmax(y) == j with high probability. That is the structural claim this
test puts a number on.

Use as a library:

    from swap_test import swap_test
    report = swap_test(W_in, W_out)
    print(report["frac_argmax_j"])

Or as a CLI:

    python swap_test.py weights/codeword_100f_50n_L4_100000steps.pt
"""
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def _to_tensor(W):
    if isinstance(W, np.ndarray):
        return torch.from_numpy(W)
    return W.detach().cpu()


def swap_test(W_in, W_out) -> Dict[str, Any]:
    """Run the swap-and-argmax test on every ordered pair (i, j) of features
    that share the same active-set size.

    Parameters
    ----------
    W_in  : (N, F) array or tensor.
    W_out : (F, N) array or tensor.

    Returns
    -------
    dict with:
      n_features, n_neurons,
      natural_argmax_correct  : how often argmax(W_out @ relu(W_in e_i)) == i
      groups                  : {K: [feature indices with popcount K]}
      n_pairs                 : number of (i, j) tested
      n_argmax_j, n_argmax_i  : counts where argmax(y) == j or == i
      frac_argmax_j, frac_argmax_i
      output_at_j, output_at_i : lists of y[j] and y[i] across pairs
    """
    W_in = _to_tensor(W_in)
    W_out = _to_tensor(W_out)
    N, F = W_in.shape
    assert W_out.shape == (F, N), \
        f"W_out shape {tuple(W_out.shape)} != ({F}, {N})"

    active_pos = []
    for j in range(F):
        h = torch.relu(W_in[:, j])
        active_pos.append(torch.where(h > 0)[0].tolist())

    # natural sanity check
    nat_correct = 0
    for i in range(F):
        h = torch.relu(W_in[:, i])
        y = W_out @ h
        if int(y.argmax()) == i:
            nat_correct += 1

    groups = defaultdict(list)
    for i, p in enumerate(active_pos):
        groups[len(p)].append(i)

    n_pairs = 0
    n_argmax_j = 0
    n_argmax_i = 0
    output_at_j = []
    output_at_i = []

    for k, members in groups.items():
        if len(members) < 2:
            continue
        for i in members:
            h_i = torch.relu(W_in[:, i])
            vals = h_i[active_pos[i]]   # K values in ascending neuron order
            for j in members:
                if j == i:
                    continue
                h_swap = torch.zeros(N)
                h_swap[active_pos[j]] = vals
                y = W_out @ h_swap
                top = int(y.argmax())
                n_pairs += 1
                if top == j:
                    n_argmax_j += 1
                if top == i:
                    n_argmax_i += 1
                output_at_j.append(float(y[j]))
                output_at_i.append(float(y[i]))

    return {
        "n_features": F,
        "n_neurons": N,
        "natural_argmax_correct": nat_correct,
        "groups": {int(k): list(v) for k, v in sorted(groups.items())},
        "n_pairs": n_pairs,
        "n_argmax_j": n_argmax_j,
        "n_argmax_i": n_argmax_i,
        "frac_argmax_j": (n_argmax_j / n_pairs) if n_pairs else float("nan"),
        "frac_argmax_i": (n_argmax_i / n_pairs) if n_pairs else float("nan"),
        "output_at_j": output_at_j,
        "output_at_i": output_at_i,
    }


def print_report(r: Dict[str, Any]) -> None:
    F = r["n_features"]
    print(f"sanity: natural argmax(W_out @ h_i) == i for "
          f"{r['natural_argmax_correct']}/{F} features")
    gs = r["groups"]
    print("popcount groups: " + ", ".join(f"K={k}:{len(v)}" for k, v in gs.items()))
    print(f"\ntransplant pairs (|s_i| = |s_j|): {r['n_pairs']}")
    print(f"  argmax == j (target):  {r['n_argmax_j']}/{r['n_pairs']}  "
          f"({100 * r['frac_argmax_j']:.1f}%)")
    print(f"  argmax == i (source):  {r['n_argmax_i']}/{r['n_pairs']}  "
          f"({100 * r['frac_argmax_i']:.1f}%)")
    j_arr = np.array(r["output_at_j"])
    i_arr = np.array(r["output_at_i"])
    if len(j_arr):
        print(f"\noutput at target j:  mean={j_arr.mean():.3f}  median={np.median(j_arr):.3f}")
        print(f"output at source i:  mean={i_arr.mean():.3f}  median={np.median(i_arr):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=str,
                    help="path to a .pt with keys 'W_in' (N,F) and 'W_out' (F,N)")
    args = ap.parse_args()

    sd = torch.load(args.weights, map_location="cpu")
    W_in = sd["W_in"]
    W_out = sd["W_out"]
    r = swap_test(W_in, W_out)
    print(f"weights: {args.weights}")
    print_report(r)


if __name__ == "__main__":
    main()
