"""Binned alternatives to the sorted-encoder-columns figure (Fig 3).

Feedback (item 3): "fix the bins but measure each bin's height individually,
overlay many scatter points per bin, with the bin line as the mean. ~5-10 bins."

mode=rank (default, the sensible reading): bin over RANK (sorted position), not
value. Each column's 50 entries are sorted descending; the 50 ranks are grouped
into ~8-10 bins. For each rank-bin we scatter the ACTUAL values across all 100
columns (so a point's vertical position is its value, not a count) and draw the
bin's mean value as a horizontal line ("height"). Off-code entries keep their
true ~-0.03 value -- nothing is collapsed to zero.

mode=value: a literal histogram over value bins (count per bin). Kept for
reference but collapses the off-code entries into a near-zero bin, which hides
the negative tail -- not recommended.

    python plot_encoder_hist.py --mode rank --bins 10
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_rank(W, nbins, fig_out):
    N, F = W.shape
    S = np.sort(W, axis=0)[::-1, :]                  # (N, F) descending per column
    edges = np.linspace(0, N, nbins + 1).astype(int)  # rank-bin boundaries

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rng = np.random.default_rng(0)
    means = []
    for b in range(nbins):
        r0, r1 = edges[b], edges[b + 1]
        vals = S[r0:r1, :].flatten()                 # actual values in this rank-bin
        xc = 0.5 * (r0 + r1)
        w = (r1 - r0)
        jit = rng.uniform(-w * 0.35, w * 0.35, size=vals.size)
        ax.plot(xc + jit, vals, "o", ms=2.0, color="#3070b8", alpha=0.12, zorder=1)
        m = vals.mean(); means.append((xc, w, m))
        ax.plot([xc - w * 0.45, xc + w * 0.45], [m, m], color="#c8412c", lw=2.4,
                zorder=3, solid_capstyle="round")
    ax.plot([], [], color="#c8412c", lw=2.4, label="bin mean")             # legend proxy
    ax.plot([], [], "o", ms=5, color="#3070b8", alpha=0.5, label="entry values")
    ax.axhline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel(f"entry rank in column (grouped into {nbins} bins)")
    ax.set_ylabel("encoder entry value")
    ax.set_xlim(0, N)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_out, dpi=130)
    print("rank-bin means (rank-center, width, mean value):")
    for xc, w, m in means:
        print(f"  ranks ~{xc:5.1f} (w={w}):  mean value = {m:+.4f}")
    print(f"saved -> {fig_out}")


def plot_value(W, nbins, fig_out, logy=False):
    N, F = W.shape
    edges = np.linspace(W.min(), W.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]); width = edges[1] - edges[0]
    counts = np.stack([np.histogram(W[:, j], bins=edges)[0] for j in range(F)])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(centers, counts.mean(0), width=width * 0.9, color="#3070b8", alpha=0.30,
           label="mean count across features")
    rng = np.random.default_rng(0)
    for b in range(nbins):
        jit = rng.uniform(-width * 0.28, width * 0.28, size=F)
        ax.plot(centers[b] + jit, counts[:, b], "o", ms=2.5, color="#c8412c", alpha=0.25)
    if logy:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.axvline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("encoder entry value"); ax.set_ylabel("count per feature column")
    ax.legend(loc="upper right", fontsize=10); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(fig_out, dpi=130)
    print(f"saved -> {fig_out}")


def plot_valuesplit(W, nbins, fig_out):
    """Value-binned count histogram with a BROKEN y-axis so both modes show.

    x = encoder entry value (nbins bins). For each feature column we count how
    many of its 50 entries fall in each bin; the bar height = mean count across
    the 100 columns and the scatter = the 100 per-column counts (jittered). The
    near-zero (off-code) bin sits at ~44 while the on-code bins sit at ~3, so a
    single linear axis hides one mode -- we split the axis into a high panel
    (off-code peak) and a low panel (on-code peak)."""
    N, F = W.shape
    edges = np.linspace(W.min(), W.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]); width = edges[1] - edges[0]
    counts = np.stack([np.histogram(W[:, j], bins=edges)[0] for j in range(F)])  # (F, nbins)
    mean_ct = counts.mean(0)

    peak = int(np.argmax(mean_ct))                          # off-code (near-zero) bin
    hi_max = mean_ct[peak]                                  # off-code peak (~44)
    lo_counts = np.delete(counts, peak, axis=1)             # exclude off-code bin
    lo_max = max(lo_counts.max() if lo_counts.size else 1, 1) * 1.25  # on-code range

    fig, (axt, axb) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 5.4),
                                   gridspec_kw={"height_ratios": [1, 2.2]})
    rng = np.random.default_rng(0)
    for ax in (axt, axb):
        ax.bar(centers, mean_ct, width=width * 0.9, color="#3070b8", alpha=0.30,
               label="mean count across features")
        for b in range(nbins):
            jit = rng.uniform(-width * 0.28, width * 0.28, size=F)
            ax.plot(centers[b] + jit, counts[:, b], "o", ms=2.5,
                    color="#c8412c", alpha=0.25)
        ax.axvline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.grid(alpha=0.3, axis="y")

    axt.set_ylim(np.floor(hi_max) - 3, np.ceil(hi_max) + 2)   # off-code peak panel
    axb.set_ylim(0, lo_max)                                   # on-code panel
    axt.spines["bottom"].set_visible(False); axb.spines["top"].set_visible(False)
    axt.tick_params(bottom=False)
    # diagonal break marks
    d = 0.012
    for (ax, ys) in [(axt, (-d, +d)), (axb, (1 - d, 1 + d))]:
        kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
        ax.plot((-0.004, 0.004), ys, **kw); ax.plot((1 - 0.004, 1 + 0.004), ys, **kw)
    axt.legend(loc="upper right", fontsize=10, framealpha=0.95)
    axb.set_xlabel("encoder entry value")
    fig.supylabel("count per feature column (50 entries each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=130)
    print("value-bin mean counts (center, mean count across features):")
    for c, m in zip(centers, mean_ct):
        print(f"  value ~{c:+.3f}:  mean count = {m:6.2f}")
    print(f"saved -> {fig_out}")


def plot_strip(W, nbins, fig_out):
    """Strip / dot plot: every encoder entry placed at its value on x.

    All N*F entries are scattered with x = entry value and y = uniform jitter
    (y carries no meaning, it only separates overlapping dots). The dot DENSITY
    reveals the bimodal structure directly: a dense band of off-code entries
    near -0.03 and a sparser band of on-code entries near +0.3. For each value
    bin we draw a vertical line at the mean value of the entries in that bin."""
    N, F = W.shape
    v = W.flatten()
    edges = np.linspace(v.min(), v.max(), nbins + 1)
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, size=v.size)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(v, y, "o", ms=2.0, color="#3070b8", alpha=0.10, zorder=1)
    means = []
    for b in range(nbins):
        m = (v >= edges[b]) & (v < edges[b + 1] if b < nbins - 1 else v <= edges[b + 1])
        if not m.any():
            continue
        mv = v[m].mean(); means.append((mv, m.sum()))
        ax.plot([mv, mv], [0.32, 0.68], color="#c8412c", lw=2.4, alpha=0.9, zorder=3,
                solid_capstyle="round")
    ax.plot([], [], color="#c8412c", lw=2.4, label="bin mean value")
    ax.plot([], [], "o", ms=5, color="#3070b8", alpha=0.5, label="encoder entry")
    ax.axvline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("encoder entry value")
    ax.set_ylabel("jitter (no meaning)")
    ax.set_yticks([])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=130)
    print("strip bin means (mean value, n entries):")
    for mv, n in means:
        print(f"  mean value = {mv:+.4f}   n = {n}")
    print(f"saved -> {fig_out}")


def plot_valuebox(W, nbins, fig_out):
    """Value-binned histogram that shows bimodality AND per-feature homogeneity.

    x = encoder entry value (nbins bins). For each feature column we count how
    many of its 50 entries fall in each bin -> a (F, nbins) count matrix. The bar
    height = mean count across the 100 features (the bimodal SHAPE). On each bar
    we annotate the ACROSS-FEATURE spread: a thick error bar = +/-1 std and thin
    caps = full min..max range. A tight annotation means every feature behaves
    the same in that bin -- e.g. the off-code bin is 44.4 +/- 0.7 (range 42-45),
    so all features are silent on ~44 neurons. That tightness is what a pooled
    histogram cannot show and is why this can stand in for the line plot.

    Broken y-axis: the off-code peak (~44) and the on-code hump (~2-3) live on
    very different scales, so we split into a high panel and a low panel."""
    N, F = W.shape
    edges = np.linspace(W.min(), W.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]); width = edges[1] - edges[0]
    counts = np.stack([np.histogram(W[:, j], bins=edges)[0] for j in range(F)])  # (F, nbins)
    mean_ct = counts.mean(0); std_ct = counts.std(0)
    lo = counts.min(0).astype(float); hi = counts.max(0).astype(float)

    # x-position of each bar's marker = the MEAN VALUE of the entries in the bin,
    # NOT the bin midpoint. The off-code bin is wide (edges -0.029..+0.049) so its
    # midpoint is +0.010 and would read as positive, although its entries average
    # ~-0.017. Placing the marker at the true mean puts it left of zero, correctly.
    vflat = W.flatten()
    binidx = np.clip(np.digitize(vflat, edges) - 1, 0, nbins - 1)
    bin_val_mean = np.array([vflat[binidx == b].mean() if (binidx == b).any()
                             else centers[b] for b in range(nbins)])

    peak = int(np.argmax(mean_ct))                         # off-code (near-zero) bin
    hi_peak = mean_ct[peak]
    on = np.delete(hi, peak)
    lo_max = max(on.max() if on.size else 1, 1) * 1.25     # on-code panel range

    fig, (axt, axb) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 5.6),
                                   gridspec_kw={"height_ratios": [1, 2.4]})

    def draw(ax):
        ax.bar(edges[:-1], mean_ct, width=width, align="edge", color="#3070b8",
               alpha=0.28, edgecolor="white", linewidth=0.5,
               label="mean count across features (bar spans bin)")
        ax.vlines(bin_val_mean, lo, hi, color="#555555", lw=1.0, alpha=0.7, zorder=2)  # range
        for c, l, h in zip(bin_val_mean, lo, hi):                                      # range caps
            ax.plot([c - width * 0.06, c + width * 0.06], [l, l], color="#555555", lw=1.0, alpha=0.7)
            ax.plot([c - width * 0.06, c + width * 0.06], [h, h], color="#555555", lw=1.0, alpha=0.7)
        ax.errorbar(bin_val_mean, mean_ct, yerr=std_ct, fmt="o", ms=4, color="#c8412c",
                    ecolor="#c8412c", elinewidth=2.6, capsize=4, zorder=3)
        ax.axvline(0.0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.grid(alpha=0.3, axis="y")

    draw(axt); draw(axb)
    axt.set_ylim(np.floor(hi_peak) - 4, np.ceil(hi.max()) + 1)   # off-code peak panel
    axb.set_ylim(0, lo_max)                                      # on-code panel
    axt.spines["bottom"].set_visible(False); axb.spines["top"].set_visible(False)
    axt.tick_params(bottom=False)
    d = 0.012
    for (ax, ys) in [(axt, (-d, +d)), (axb, (1 - d, 1 + d))]:
        kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
        ax.plot((-0.004, 0.004), ys, **kw); ax.plot((1 - 0.004, 1 + 0.004), ys, **kw)

    # legend proxies for the spread annotation
    axt.errorbar([], [], yerr=[], fmt="o", ms=4, color="#c8412c", ecolor="#c8412c",
                 elinewidth=2.6, capsize=4, label="mean ± 1 std across features")
    axt.plot([], [], color="#555555", lw=1.0, label="min–max across features")
    axt.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    axb.set_xlabel("encoder entry value")
    fig.supylabel("count per feature column (50 entries each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=130)
    print("value-bin stats (bin range -> mean value of entries: count mean +/- std [min, max]):")
    for b in range(nbins):
        print(f"  [{edges[b]:+.3f},{edges[b+1]:+.3f}] mean_val={bin_val_mean[b]:+.4f}: "
              f"{mean_ct[b]:6.2f} +/- {std_ct[b]:4.2f}  [{int(lo[b])}, {int(hi[b])}]")
    print(f"saved -> {fig_out}")


def plot_valuedots(W, nbins, fig_out, scale="linear", spread=True, barfrac=1.0,
                   step=0.05):
    """Pooled-count value histogram with bin edges aligned to zero.

    x = encoder entry value, binned in fixed steps of ``step`` with edges aligned
    to 0 (so a bin boundary sits exactly at zero, separating negatives from
    positives), each bin spanning ``step``. Bar HEIGHT = count of entries in the
    bin (pooled across all 100 features); bars are drawn at their true bin width
    (``barfrac=1.0``). The actual entries are also scattered at x = their true
    value so the real distribution is visible. No bin-mean line.

    scale: "linear" (single axis -- on-code looks small but honest, default),
           "break"  (broken y so off-code ~4400 and on-code ~200-300 both show),
           "log"    (single log axis).
    spread: True scatters each entry at a random height filling its bar; False
            collapses the scatter to a thin rug at the base.
    barfrac: bar width as a fraction of the bin width (1.0 = bars are the bins)."""
    N, F = W.shape
    v = W.flatten()
    # bin edges aligned to 0 (a bin boundary sits exactly at zero); width = step
    e0 = int(np.floor(v.min() / step))
    e1 = int(np.ceil(v.max() / step))
    edges = np.arange(e0, e1 + 1) * step
    centers = (edges[:-1] + edges[1:]) / 2
    nb = len(centers)
    n_b, _ = np.histogram(v, bins=edges)
    binidx = np.clip(np.digitize(v, edges) - 1, 0, nb - 1)

    log = (scale == "log")
    base = 0.7 if log else 0.0
    rng = np.random.default_rng(0)
    cnt = np.maximum(n_b[binidx], 1)
    if not spread:
        yj = np.full(v.size, (base * 1.3 + 1.0) if log else 0.0)
    elif log:
        yj = cnt.astype(float) ** rng.uniform(0, 1, size=v.size)
    else:
        yj = rng.uniform(0, cnt)

    def draw(ax):
        ax.bar(centers, n_b - base, width=step * barfrac, bottom=base,
               color="#cdddef", alpha=0.7, edgecolor="#3070b8", linewidth=0.8,
               zorder=1, label="Count of entries in bin")
        ms = 1.6 if spread else 2.2
        ax.plot(v, yj, "o", ms=ms, color="#3070b8", alpha=0.09, zorder=2)
        ax.grid(alpha=0.3, axis="y")

    if scale == "break":
        peak = int(np.argmax(n_b)); lo_max = np.delete(n_b, peak).max() * 1.30
        fig, (axt, axb) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 5.8),
                                       gridspec_kw={"height_ratios": [1, 2.4]})
        draw(axt); draw(axb)
        axt.set_ylim(n_b[peak] * 0.965, n_b[peak] * 1.02)
        axb.set_ylim(0, lo_max)
        axt.spines["bottom"].set_visible(False); axb.spines["top"].set_visible(False)
        axt.tick_params(bottom=False)
        d = 0.012
        for (ax, ys) in [(axt, (-d, +d)), (axb, (1 - d, 1 + d))]:
            kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
            ax.plot((-0.004, 0.004), ys, **kw); ax.plot((1 - 0.004, 1 + 0.004), ys, **kw)
        legax, xax = axt, axb
    else:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        draw(ax)
        if log:
            ax.set_yscale("log"); ax.set_ylim(base, n_b.max() * 1.6)
        else:
            ax.set_ylim(0, n_b.max() * 1.06)
        legax, xax = ax, ax

    legax.plot([], [], "o", ms=5, color="#3070b8", alpha=0.5,
               label="Encoder entry (true value)")
    legax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    t0 = np.floor(edges[0] / 0.1) * 0.1
    xax.set_xticks(np.round(np.arange(t0, edges[-1] + 1e-9 + 0.1, 0.1), 1))
    xax.set_xlabel("Encoder entry value")
    fig.supylabel("Count of encoder entries (pooled over all features)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=130)
    print(f"[scale={scale} spread={spread}] decimal-centered pooled counts:")
    for c, n in zip(centers, n_b):
        print(f"  value ~{c:+.1f}: n={n}")
    print(f"saved -> {fig_out}")


def plot_kdist(W, fig_out, tau=0.05, axis="col"):
    """Histogram of the number of LARGE entries per column (or row).

    Companion to the value histogram. The value histogram shows the bimodal
    value distribution but averages over columns; this one shows that the
    columns are all alike: K = (W > tau).sum over the chosen axis, then histogram
    K across features (axis='col') or neurons (axis='row'). A tight spike means
    every column/row uses essentially the same number of large entries."""
    from matplotlib.patches import Patch
    N, F = W.shape
    BLUE, RED = "#3070b8", "#c8412c"
    Kc, Kr = (W > tau).sum(0), (W > tau).sum(1)             # per column, per row

    def bars(ax, K, color, ntot):
        centers = np.arange(K.min(), K.max() + 1)
        counts = np.array([(K == c).sum() for c in centers])
        ax.bar(centers, counts, width=0.60, color=color, alpha=0.55,
               edgecolor=color, linewidth=1.0)
        ax.set_ylim(0, ntot)
        print(f"large-entry count (tau={tau}): "
              + "  ".join(f"K={c}:{n}" for c, n in zip(centers, counts) if n)
              + f"  mean={K.mean():.2f} std={K.std():.2f} range=[{K.min()},{K.max()}]")
        return centers

    if axis == "both":
        fig, axL = plt.subplots(figsize=(7.0, 4.4))
        axR = axL.twinx()
        cc = bars(axL, Kc, BLUE, F)
        rc = bars(axR, Kr, RED, N)
        axL.set_ylabel(r"Count of features with $n$ entries", color=BLUE)
        axR.set_ylabel(r"Count of neurons with $n$ entries", color=RED)
        axL.tick_params(axis="y", colors=BLUE); axR.tick_params(axis="y", colors=RED)
        axL.spines["left"].set_color(BLUE); axR.spines["right"].set_color(RED)
        ticks = np.arange(min(cc.min(), rc.min()), max(cc.max(), rc.max()) + 1)
        handles = [Patch(facecolor=BLUE, alpha=0.55, edgecolor=BLUE,
                         label=f"Per feature column:  {Kc.mean():.1f} ± {Kc.std():.1f}"),
                   Patch(facecolor=RED, alpha=0.55, edgecolor=RED,
                         label=f"Per neuron row:  {Kr.mean():.1f} ± {Kr.std():.1f}")]
        axL.legend(handles=handles, loc="upper center", fontsize=9.5,
                   framealpha=0.95, title="Encoder large-entry count")
        xax = axL
    else:
        K, color, lab, ntot = ((Kc, BLUE, "feature column", F) if axis == "col"
                               else (Kr, RED, "neuron row", N))
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ticks = bars(ax, K, color, ntot)
        ax.set_ylabel(rf"Count of {lab}s with $n$ entries")
        ax.legend(handles=[Patch(facecolor=color, alpha=0.55, edgecolor=color,
                                 label=f"mean = {K.mean():.2f} ± {K.std():.2f}")],
                  loc="upper right", fontsize=10, framealpha=0.95)
        xax = ax

    xax.set_xticks(ticks)
    xax.set_xlabel(rf"Number of large entries $n$  ($w > {tau}$)")
    xax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_out, dpi=130)
    print(f"saved -> {fig_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str,
                    default="weights/embedded_100f_50n_d1000_L4_100000steps.pt")
    ap.add_argument("--mode",
                    choices=["rank", "value", "valuesplit", "valuebox",
                             "valuedots", "kdist", "strip"],
                    default="rank")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="kdist mode: large-entry threshold")
    ap.add_argument("--axis", choices=["col", "row", "both"], default="col",
                    help="kdist mode: count large entries per column, row, or both")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--logy", action="store_true", help="value mode only")
    ap.add_argument("--scale", choices=["break", "linear", "log"], default="break",
                    help="valuedots mode: y-axis style")
    ap.add_argument("--nospread", action="store_true",
                    help="valuedots mode: collapse scatter to a base rug")
    ap.add_argument("--barfrac", type=float, default=1.0,
                    help="valuedots mode: bar width as fraction of bin width")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sd = torch.load(args.weights, map_location="cpu")
    W = (sd["W_in_eff"] if "W_in_eff" in sd else sd["W_in"]).numpy()
    suffix = "_embedded" if "W_in_eff" in sd else ""
    fig_out = Path(args.out) if args.out else Path(
        f"figures/encoder_hist_{args.mode}{suffix}.png")
    fig_out.parent.mkdir(exist_ok=True)
    if args.mode == "rank":
        plot_rank(W, args.bins, fig_out)
    elif args.mode == "valuesplit":
        plot_valuesplit(W, args.bins, fig_out)
    elif args.mode == "valuebox":
        plot_valuebox(W, args.bins, fig_out)
    elif args.mode == "valuedots":
        plot_valuedots(W, args.bins, fig_out, scale=args.scale,
                       spread=not args.nospread, barfrac=args.barfrac)
    elif args.mode == "kdist":
        plot_kdist(W, fig_out, tau=args.tau, axis=args.axis)
    elif args.mode == "strip":
        plot_strip(W, args.bins, fig_out)
    else:
        plot_value(W, args.bins, fig_out, logy=args.logy)


if __name__ == "__main__":
    main()
