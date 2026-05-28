"""Plot L4 loss vs codeword length K for birregular and random codes.

Reads results produced by ``sweep_K_codes.py``.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/sweep_K_codes.json")
OUT = Path("figures/loss_vs_K_codes.png")


def main():
    with open(DATA) as f:
        res = json.load(f)

    K_list = res["meta"]["K_list"]
    l4_trained = res["meta"]["l4_trained_100k"]
    swaps = res["meta"].get("swaps", None)

    def stats(code):
        means, stds, all_pts = [], [], []
        for K in K_list:
            ls = np.array(res[code][str(K)]["losses"])
            means.append(ls.mean())
            stds.append(ls.std(ddof=1))
            all_pts.append(ls)
        return np.array(means), np.array(stds), all_pts

    reg_mu, reg_sd, reg_pts = stats("regular")
    rnd_mu, rnd_sd, rnd_pts = stats("random")

    # express everything as a fraction of the trained model's L4 loss
    reg_mu, reg_sd = reg_mu / l4_trained, reg_sd / l4_trained
    rnd_mu, rnd_sd = rnd_mu / l4_trained, rnd_sd / l4_trained
    reg_pts = [p / l4_trained for p in reg_pts]
    rnd_pts = [p / l4_trained for p in rnd_pts]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.errorbar(K_list, reg_mu, yerr=reg_sd, color="#1f77b4", lw=1.8,
                marker="o", markersize=7, capsize=4, label="birregular code")
    ax.errorbar(K_list, rnd_mu, yerr=rnd_sd, color="#d62728", lw=1.8,
                marker="s", markersize=7, capsize=4, label="random code")

    rng = np.random.default_rng(0)
    for K, pts in zip(K_list, reg_pts):
        jit = rng.uniform(-0.08, 0.08, size=len(pts))
        ax.plot(np.full_like(pts, K) + jit, pts, "o", color="#1f77b4",
                markersize=3.5, alpha=0.4)
    for K, pts in zip(K_list, rnd_pts):
        jit = rng.uniform(-0.08, 0.08, size=len(pts))
        ax.plot(np.full_like(pts, K) + jit, pts, "s", color="#d62728",
                markersize=3.5, alpha=0.4)

    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1.4,
               label=f"trained 100k model  (L4={l4_trained:.2e})")

    ax.set_xticks(K_list)
    ax.set_xlabel("codeword length K (active neurons per feature)")
    ax.set_ylabel(r"$L^4$ loss  /  trained-model $L^4$ loss")
    swap_str = f", {swaps // 1000}k overlap swaps" if swaps else ""
    ax.set_title(
        f"L4 vs codeword length: birregular vs random  "
        f"(F={res['meta']['F']}, N={res['meta']['N']}, p={res['meta']['P']}, "
        f"5 seeds each{swap_str})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")
    print()
    print("summary (mean L4 +/- std over 5 seeds):")
    print(f"  {'K':>3}  {'regular':>20}  {'random':>20}  {'rand/reg':>9}")
    for K, rm, rs, nm, ns in zip(K_list, reg_mu, reg_sd, rnd_mu, rnd_sd):
        print(f"  {K:>3}  {rm:>10.3e} +/- {rs:>6.0e}  {nm:>10.3e} +/- {ns:>6.0e}  "
              f"{nm / rm:>8.2f}x")


if __name__ == "__main__":
    main()
