"""Input-output response of the trained network (the attenuation figure).

Following Bhagat et al.'s left plot: drive one feature j with a single value v
(all other features zero) and read the model's output at index j. With no
interference present, an un-attenuated network would trace y = ReLU(v) exactly.
The trained network instead outputs a *scaled-down* ReLU -- it attenuates the
active output. The reason is interference at the typical operating point
(p=0.02 => ~2 active features): when several features are active their
distributed codewords overlap and leak into inactive outputs, so the network
learns to shrink each active output to keep the larger multi-feature error
(which L^4 punishes heavily) down. Driving a single feature in isolation still
shows this learned attenuation.

  python plot_io_response.py

Saves figures/io_response.png.
"""
from pathlib import Path

import numpy as np
import torch

from small_models import DEVICE

CKPT = "weights/codeword_100f_50n_L4_100000steps.pt"
F, N = 100, 50


def main():
    sd = torch.load(CKPT, map_location=DEVICE)
    W_in = sd["W_in"].to(DEVICE)        # (N, F)
    W_out = sd["W_out"].to(DEVICE)      # (F, N)

    vs = torch.linspace(-1, 1, 201, device=DEVICE)
    # For each feature j, build inputs x = v * e_j and read output[j].
    # diag_resp[j, t] = yhat_j when feature j is driven with vs[t].
    diag = torch.zeros(F, vs.numel(), device=DEVICE)
    with torch.no_grad():
        for j in range(F):
            x = torch.zeros(vs.numel(), F, device=DEVICE)
            x[:, j] = vs
            yhat = torch.relu(x @ W_in.T) @ W_out.T
            diag[j] = yhat[:, j]
    diag = diag.cpu().numpy()
    vs_np = vs.cpu().numpy()

    mean_resp = diag.mean(0)
    lo, hi = np.percentile(diag, [5, 95], axis=0)

    # slope of the mean response on the active branch (v>0), as an attenuation factor
    pos = vs_np > 0
    slope = np.polyfit(vs_np[pos], mean_resp[pos], 1)[0]
    print(f"mean active-branch slope (attenuation factor) ~ {slope:.3f}")
    print(f"(1.0 = no attenuation; <1 = output shrunk relative to true ReLU)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(vs_np, np.maximum(vs_np, 0), color="#444", lw=1.6,
            ls=(0, (6, 4)), label="true ReLU (no attenuation)")
    ax.fill_between(vs_np, lo, hi, color="#3070b8", alpha=0.2,
                    label="5-95% across features")
    ax.plot(vs_np, mean_resp, color="#3070b8", lw=2.0,
            label=f"trained output (mean, slope {slope:.2f})")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("input value of the active feature, $v$")
    ax.set_ylabel("model output at that feature, $\\hat y_j$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("figures/io_response.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
