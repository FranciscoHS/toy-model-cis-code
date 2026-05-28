"""Binary codeword construction and overlap reduction.

The same code-generation primitives are used by ``sweep_K_codes.py`` (to fit
the 3-parameter model at each codeword length) and by ``plot_W_in_heatmaps.py``
(to compare the trained W_in to a clean birregular ansatz).

A *code* is a binary matrix M of shape (F, N): row i is the codeword for
feature i, with K=row_sum(i) active neurons.
"""
import numpy as np


def regular_code(n_features, n_neurons, K, seed):
    """Birregular bipartite code: each feature has degree K, each neuron has
    degree d = F*K/N. Built via stub-matching with local repair to remove
    multi-edges. ``F*K`` must be divisible by ``N``.
    """
    F, N = n_features, n_neurons
    assert F * K % N == 0, f"F*K must be divisible by N (F={F}, N={N}, K={K})"
    d = F * K // N
    rng = np.random.default_rng(seed)
    fstubs = np.repeat(np.arange(F), K)
    nstubs = rng.permutation(np.repeat(np.arange(N), d))
    M = np.zeros((F, N), int)
    for f, n in zip(fstubs, nstubs):
        M[f, n] += 1
    for _ in range(200_000):
        dup = np.argwhere(M > 1)
        if len(dup) == 0:
            break
        f, n = dup[0]
        cand = np.argwhere(M == 1)
        rng.shuffle(cand)
        for f2, n2 in cand:
            if f2 != f and M[f, n2] == 0 and M[f2, n] == 0:
                M[f, n] -= 1
                M[f2, n2] -= 1
                M[f, n2] += 1
                M[f2, n] += 1
                break
        else:
            raise RuntimeError("birregular repair stuck")
    return M.astype(bool)


def random_code(n_features, n_neurons, K, seed):
    """Each feature picks K random neurons (no neuron-degree constraint)."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n_features, n_neurons), dtype=bool)
    for f in range(n_features):
        cols = rng.choice(n_neurons, K, replace=False)
        M[f, cols] = True
    return M


def reduce_overlap(M, iters, seed):
    """Edge swaps that lower row-pair overlap variance, preserving row and
    column sums. Keeps a birregular code birregular; for a random code it
    decreases ``sum(overlap^2)`` without enforcing a uniform column degree.
    """
    rng = np.random.default_rng(seed)
    M = M.copy()
    edges = list(zip(*np.where(M)))
    ov = M.astype(int) @ M.astype(int).T
    np.fill_diagonal(ov, 0)
    obj = (ov ** 2).sum()
    for _ in range(iters):
        (i, a), (j, b) = (
            edges[rng.integers(len(edges))],
            edges[rng.integers(len(edges))],
        )
        if i == j or a == b or M[i, b] or M[j, a]:
            continue
        M[i, a] = M[j, b] = False
        M[i, b] = M[j, a] = True
        ni = M.astype(int) @ M[i].astype(int)
        ni[i] = 0
        nj = M.astype(int) @ M[j].astype(int)
        nj[j] = 0
        new_obj = (
            obj
            - (ov[i] ** 2).sum()
            - (ov[j] ** 2).sum()
            + (ni ** 2).sum()
            + (nj ** 2).sum()
            - 2 * (ov[i, j] ** 2)
            + 2 * (ni[j] ** 2)
        )
        if new_obj <= obj:
            obj = new_obj
            ov[i] = ov[:, i] = ni
            ov[j] = ov[:, j] = nj
            ov[i, i] = ov[j, j] = 0
            edges = list(zip(*np.where(M)))
        else:
            M[i, a] = M[j, b] = True
            M[i, b] = M[j, a] = False
    return M
