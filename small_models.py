"""Shared model + data primitives for the F=100, N=50, p=0.02 toy model.

The trained checkpoints in weights/ are state_dicts with keys ``W_in`` and
``W_out`` matching the parameters on :class:`SimpleMLP`, so they load with
``model.load_state_dict(torch.load(path))`` and also with a plain
``torch.load(path)`` if you only want the raw tensors.
"""
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleMLP(nn.Module):
    """Single hidden layer: y = W_out @ relu(W_in @ x), no biases."""

    def __init__(self, n_features, n_neurons):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_features, n_neurons) * 0.01)

    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T


def generate_batch(batch_size, n_features, p, device=DEVICE):
    """Sparse Bernoulli-Uniform features with target y = relu(x).

    Each entry is active w.p. ``p``; active values are uniform on (-1, 1).
    """
    mask = (torch.rand(batch_size, n_features, device=device) < p).float()
    values = torch.rand(batch_size, n_features, device=device) * 2 - 1
    x = mask * values
    y = torch.relu(x)
    return x, y


def evaluate_per_feature_mse(W_in, W_out, n_features, p, n_batches=100,
                             batch_size=2048, seed=9999, device=DEVICE):
    """Per-feature MSE conditioned on feature being active.

    Matches the convention used in the per-feature loss plot.
    """
    torch.manual_seed(seed)
    err_sum = torch.zeros(n_features, device=device)
    cnt = torch.zeros(n_features, device=device)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = generate_batch(batch_size, n_features, p, device=device)
            yhat = torch.relu(x @ W_in.T) @ W_out.T
            active = (x > 0).float()
            err_sum += ((yhat - y) ** 2 * active).sum(dim=0)
            cnt += active.sum(dim=0)
    return (err_sum / cnt.clamp_min(1)).cpu().numpy()
