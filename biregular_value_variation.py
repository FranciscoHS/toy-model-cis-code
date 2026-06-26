"""Do on-code encoder values still vary under a *designed biregular* code?

Stefan's question re: the trained network's r~-0.6 between neuron usage and
on-code value. In a perfectly biregular code every neuron has the same degree,
so that particular driver of variation is removed. If on-code values then
collapse to (near-)uniform, the trained net's value spread is largely explained
by its unequal neuron loads (which arise because, e.g., F*K/N isn't a clean
integer for the trained K-distribution). If they still vary a lot, something
else is going on (e.g. per-pair overlap structure).

We:
  1. measure the spread (CV) of on-code values in the TRAINED no-embed net, and
     their correlation with neuron degree;
  2. take a designed biregular K=5 code (with edge swaps), fit FREE per-edge
     on-code values (off-code = single scalar b, decoder = c*pinv), and measure
     the CV of the fitted on-code values + their correlation with per-edge
     overlap load.

  python biregular_value_variation.py
"""
import numpy as np
import torch

from small_models import generate_batch, DEVICE
from codes import regular_code, reduce_overlap

F, N, P, TAU = 100, 50, 0.02, 0.05
K = 5
SWAPS = 800_000
STEPS = 15_000


def trained_stats():
    sd = torch.load("weights/codeword_100f_50n_L4_100000steps.pt", map_location="cpu")
    W_in = sd["W_in"].numpy()                  # (N, F)
    M = W_in.T > TAU                           # (F, N) support
    on_vals = W_in.T[M]                        # on-code values
    neuron_deg = M.sum(0)                      # codewords per neuron, (N,)
    # value vs degree of the neuron each on-edge sits on
    edge_neuron = np.broadcast_to(np.arange(N), M.shape)[M]
    deg_per_edge = neuron_deg[edge_neuron]
    r_edge = np.corrcoef(deg_per_edge, on_vals)[0, 1]
    # per-neuron: mean on-value of each neuron's incoming edges vs its degree
    W = W_in.T                                  # (F, N)
    neuron_mean_onval = np.array([W[M[:, n], n].mean() for n in range(N)])
    r_neuron = np.corrcoef(neuron_deg, neuron_mean_onval)[0, 1]
    return dict(cv=on_vals.std() / on_vals.mean(),
                mean=on_vals.mean(), std=on_vals.std(),
                r_value_vs_degree_edge=r_edge,
                r_value_vs_degree_neuron=r_neuron,
                deg_range=(neuron_deg.min(), neuron_deg.max()))


def fit_biregular():
    M = regular_code(F, N, K, seed=0)
    M = reduce_overlap(M, SWAPS, seed=10)
    neuron_deg = M.sum(0)
    M_NF = torch.tensor(M.T.astype(np.float32), device=DEVICE)   # (N, F)

    torch.manual_seed(31337)
    x_tr, y_tr = generate_batch(600_000, F, P)
    x_ev, y_ev = generate_batch(600_000, F, P)

    # free per-entry on-code values V (only support entries matter), scalar off b, scale c
    V = (torch.full((N, F), 0.9, device=DEVICE)).requires_grad_()
    b = torch.tensor(-0.04, device=DEVICE, requires_grad=True)
    c = torch.tensor(1.76, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([V, b, c], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for _ in range(STEPS):
        idx = torch.randint(0, x_tr.shape[0], (8192,), device=DEVICE)
        Wi = V * M_NF + b * (1 - M_NF)
        Wo = c * torch.linalg.pinv(Wi)
        loss = ((torch.relu(x_tr[idx] @ Wi.T) @ Wo.T - y_tr[idx]).abs() ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        Wi = V * M_NF + b * (1 - M_NF)
        Wo = c * torch.linalg.pinv(Wi)
        l4 = ((torch.relu(x_ev @ Wi.T) @ Wo.T - y_ev).abs() ** 4).mean().item()
        on_vals = (V * M_NF).cpu().numpy().T[M]    # on-code values, (n_edges,)

    # per-edge overlap load: total overlap of the neuron's column? use per-pair.
    ov = M.astype(int) @ M.astype(int).T
    np.fill_diagonal(ov, 0)
    # load on a neuron = sum of overlaps of features sharing it -> approximate via column
    edge_neuron = np.broadcast_to(np.arange(N), M.shape)[M]
    # neuron "busyness" proxy: how much total pairwise overlap its features incur
    neuron_overlap_load = np.array([ov[np.where(M[:, n])[0]].sum() for n in range(N)])
    r_ov = np.corrcoef(neuron_overlap_load[edge_neuron], on_vals)[0, 1]
    return dict(cv=on_vals.std() / on_vals.mean(), mean=on_vals.mean(),
                std=on_vals.std(), l4_ratio=None, l4=l4,
                deg_range=(int(neuron_deg.min()), int(neuron_deg.max())),
                r_value_vs_overlap=r_ov)


def main():
    print(f"device: {DEVICE}\n")
    t = trained_stats()
    print("TRAINED net on-code values:")
    print(f"  CV = {t['cv']:.3f}  (mean {t['mean']:.3f}, std {t['std']:.3f})")
    print(f"  neuron degree range = {t['deg_range']}  (unequal loads)")
    print(f"  corr(on-value, neuron degree): per-edge = {t['r_value_vs_degree_edge']:.3f}  "
          f"per-neuron = {t['r_value_vs_degree_neuron']:.3f}\n")

    b = fit_biregular()
    print("DESIGNED biregular K=5 (free per-edge on-values):")
    print(f"  L4 = {b['l4']:.4e}")
    print(f"  CV = {b['cv']:.3f}  (mean {b['mean']:.3f}, std {b['std']:.3f})")
    print(f"  neuron degree range = {b['deg_range']}  (equal loads by construction)")
    print(f"  corr(on-value, neuron overlap load) = {b['r_value_vs_overlap']:.3f}\n")

    print(f"=> on-code value CV: trained {t['cv']:.3f}  vs  biregular {b['cv']:.3f}")
    if b['cv'] < 0.5 * t['cv']:
        print("   biregular values are much more uniform -> unequal neuron load "
              "explains most of the trained net's value spread.")
    else:
        print("   biregular values still vary substantially -> load imbalance "
              "is not the whole story.")


if __name__ == "__main__":
    main()
