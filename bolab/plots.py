"""Figures for the README (static PNG, validated categorical palette, direct labels)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e2dc", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow (validated, light)
NEUTRAL = "#b4b2aa"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#ece9e2", "#2a78d6"])

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal"})


def _style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def sharpe_distribution(sr_real, sr_plac, dsr, best, textbook_sr, out):
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=160)
    bins = np.arange(-0.85, 0.6, 0.05)
    ax.hist(sr_plac, bins=bins, color=NEUTRAL, rwidth=0.88, label="512 placebo backtests (random signals)")
    ax.hist(sr_real, bins=bins, color=SERIES[0], rwidth=0.88, alpha=0.9, label="512 momentum backtests")
    sr0 = dsr["n=512"]["sr0_annual"]
    ymax = ax.get_ylim()[1]
    for x, text, ls, y, ha in ((sr0, f"luck alone: {sr0:.2f} ", "--", 0.97, "right"),
                               (textbook_sr, f" textbook 12-1: {textbook_sr:.2f}", ":", 0.97, "left"),
                               (sr_real.max(), f" best: {sr_real.max():.2f}", "-", 0.55, "left")):
        ax.axvline(x, color=INK, lw=1, ls=ls)
        ax.text(x, ymax * y, text, rotation=90, va="top", ha=ha, fontsize=7.5, color=INK)
    ax.set_xlabel("Sharpe ratio, annualised, net of 10 bp costs (1930–2026)", color=INK2, fontsize=8)
    ax.set_ylabel("number of backtests", color=INK2, fontsize=8)
    ax.set_title("Best of 512 momentum backtests vs the best that luck alone produces in a search this size",
                 loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "sharpe_distribution.png")
    plt.close(fig)


def cscv(p, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=160, gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    sin, sout = p["sel_in"] * np.sqrt(12), p["sel_out"] * np.sqrt(12)
    ax.scatter(sin, sout, s=4, color=SERIES[0], alpha=0.12, linewidths=0)
    lo, hi = min(sin.min(), sout.min()) - 0.05, max(sin.max(), sout.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
    ax.text(hi, hi, "out = in ", ha="right", va="bottom", fontsize=7, color=INK2)
    med = np.median(p["median_out"]) * np.sqrt(12)
    ax.axhline(med, color=SERIES[1], lw=1)
    ax.text(lo + 0.02, med + 0.02, f"median of all 512 out of sample: {med:.2f}", fontsize=7, color=INK)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlabel("Sharpe of the in-sample winner, in sample", color=INK2, fontsize=8)
    ax.set_ylabel("same strategy, out of sample", color=INK2, fontsize=8)
    ax.set_title(f"{p['splits']:,} train/test splits of 1930–2026", loc="left", color=INK)
    _style(ax, grid="both")

    ax = axes[1]
    ax.hist(p["logit"], bins=40, color=SERIES[0], rwidth=0.88)
    ax.axvline(0, color=INK, lw=1)
    ax.text(0, ax.get_ylim()[1] * 0.95, f" PBO = {p['pbo']:.0%}\n winner below median\n out of sample",
            fontsize=7.5, va="top", ha="left", color=INK)
    ax.set_xlabel("logit of the winner's out-of-sample rank", color=INK2, fontsize=8)
    ax.set_ylabel("splits", color=INK2, fontsize=8)
    ax.set_title("Probability of backtest overfitting", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "cscv.png")
    plt.close(fig)


def cumulative(series: dict, out, start="1950-07"):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    for (name, s), color in zip(series.items(), SERIES):
        s = s.loc[start:].dropna()
        w = (1 + s).cumprod()
        x = s.index.to_timestamp()
        ax.plot(x, w, color=color, lw=2 if "walk" in name else 1.4)
        ax.text(x[-1], w.iloc[-1], f"  {name.split(' (')[0]}  ×{w.iloc[-1]:.0f}", va="center", fontsize=7.5,
                color=INK)
    ax.set_yscale("log")
    ax.axvspan(pd.Timestamp("2009-03-01"), pd.Timestamp("2009-06-30"), color=GRID, lw=0)
    ax.text(pd.Timestamp("2009-05-01"), 0.8, "2009 momentum crash", fontsize=7, color=INK2, ha="center")
    ax.set_ylabel("growth of 1 (log scale)", color=INK2, fontsize=8)
    ax.set_title("Re-picking the best backtest every 5 years did worse than fixed textbook momentum (1950–2026)",
                 loc="left", color=INK)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * 0.3)
    years = range(1950, 2030, 10)
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in years], [str(y) for y in years])
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "cumulative.png")
    plt.close(fig)


def landscape(sr: pd.Series, cfgs, out):
    df = pd.DataFrame([{"family": c.family, "lookback": c.lookback, "hold": c.hold, "sharpe": sr[c.name]}
                       for c in cfgs])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), dpi=160)
    lim = float(np.abs(df.sharpe).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for ax, fam, title in ((axes[0], "XS", "Cross-sectional: rank industries"),
                           (axes[1], "TS", "Time-series: each industry's own trend")):
        piv = df[df.family == fam].pivot_table(index="lookback", columns="hold", values="sharpe", aggfunc="median")
        ax.imshow(piv.to_numpy(), cmap=DIVERGING, norm=norm, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(piv.shape[1]), [f"{h}" for h in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{l}" for l in piv.index])
        ax.set_xlabel("holding period, months", color=INK2, fontsize=8)
        ax.set_ylabel("lookback, months", color=INK2, fontsize=8)
        ax.set_title(title, loc="left", color=INK)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle("Median Sharpe by lookback and holding period: a broad ridge, not a lucky isolated peak",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "landscape.png")
    plt.close(fig)


def plot_all(M, P, sr_real, sr_plac, cscv_result, series, cfgs, dsr, best, out):
    textbook_sr = float(sr_real["XS-L12-S1-H1-q30-ew"])
    sharpe_distribution(sr_real, sr_plac, dsr, best, textbook_sr, out)
    cscv(cscv_result, out)
    cumulative(series, out)
    landscape(sr_real, cfgs, out)
