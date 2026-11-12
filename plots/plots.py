"""
plots.py - figures for the report, from one profile run (data loaded by data_summary.py).

Usage (from the project root):
    .venv/bin/python plots/plots.py                                # latest run in logs/profile
    .venv/bin/python plots/plots.py logs/profile/20261104-105307   # a specific run
    .venv/bin/python plots/plots.py ... --show                     # also open the figures

Output: <run_dir>/figures/*.png (200 dpi) and *.pdf (vector, for LaTeX/Word)

Figures:
    1_speedup.*            mean wall time speedup of every binary vs run_fp32_baseline
    2_thread_scaling.*     OpenMP speedup vs 1 thread (with ideal line) and tok/s, per thread count
    3_perplexity_speed.*   perplexity (from perplexity.csv) against speedup
    4_tok_s_boxplot.*      raw tok/s of every run: (a) timing runs, (b) thread sweep; mean marked
    5_cachegrind.*         simulated I1 / D1 / LL data miss rates
    6_perf_breakdown.*     share of CPU time per function group (perf record)
    7_cachegrind_threads.* total simulated L1 / last-level data misses per OpenMP thread count
    8_model_size.*         model .bin size per precision (FP32 / FP16 / INT8) and peak memory (max RSS)
    9_next_token_agreement.*  (a) % of positions whose argmax differs from FP32,
                           (b) what those mismatches were: FP32 right / this binary right / both wrong

Colour encodes precision (FP32 / FP16 / INT8) wherever binaries are compared, the same in
every figure. Palette checked for colour-blind separation; values are printed on the marks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_summary import BASELINE, ROOT, RunData, latest_run_dir, load_run

# ----------------------------------------------------------------------------
# Style

INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text, axes
GRID = "#e5e4e0"       # recessive grid
SURFACE = "#ffffff"
NEUTRAL = "#a3a29c"    # "other" / reference lines

PRECISION_COLOURS = {"FP32": "#2a78d6", "FP16": "#eb6834", "INT8": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order, never cycled
CACHE_COLOURS = {"I1_miss_rate": "#86b6ef", "D1_miss_rate": "#2a78d6", "LL_data_miss_rate": "#104281"}

# Short names on axes (binary -> label); unknown binaries keep their name
LABELS = {
    "run_fp32_baseline": "FP32 baseline",
    "run_fp16_baseline": "FP16 baseline\n(software decode)",
    "run_fp16_baseline_native": "FP16 baseline\n(_Float16)",
    "run_int8_baseline": "INT8 baseline",
    "run_fp32_simd": "FP32 SIMD",
    "run_fp32_simd_4row": "FP32 SIMD 4-row",
    "run_fp32_openmp": "FP32 OpenMP",
    "run_fp32_openmp_simd": "FP32 OpenMP\n+ -O3 SIMD",
    "run_fp32_openmp_neon": "FP32 OpenMP\n+ NEON",
}


def label(binary) -> str:
    return LABELS.get(str(binary), str(binary))


def one_line(binary) -> str:
    return label(binary).replace("\n", " ")


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.markersize": 7,
    })


def precision_legend(ax, precisions, **kwargs) -> None:
    handles = [Patch(facecolor=PRECISION_COLOURS[p], label=p) for p in PRECISION_COLOURS if p in set(precisions)]
    ax.legend(handles=handles, title="Precision", **kwargs)


def subtitle(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=8, color=INK_2, ha="left", va="bottom")


# ----------------------------------------------------------------------------
# 1. Speedup bar chart


def plot_speedup(data: RunData):
    df = data.timing_summary.copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))[::-1]  # first binary at the top
    colours = [PRECISION_COLOURS.get(p, NEUTRAL) for p in df["precision"]]
    ax.barh(y, df["speedup_vs_baseline"], color=colours, height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.axvline(1, color=INK_2, linewidth=1, linestyle="--")
    ax.text(1, y.max() + 0.6, " baseline = 1×", color=INK_2, fontsize=8, va="bottom")

    for yi, (_, row) in zip(y, df.iterrows()):
        ax.text(max(row["speedup_vs_baseline"], 1.0), yi,  # start after the 1× line for slower binaries
                f"  {row['speedup_vs_baseline']:.2f}×  ({row['wall_s_mean']:.2f} s)",
                va="center", ha="left", fontsize=9, color=INK)
    ticks = [f"{label(b)}  [{t} thr]" if t > 1 else label(b) for b, t in zip(df["binary"], df["threads"])]
    ax.set_yticks(y, ticks)
    ax.set_xlabel(f"Speedup vs {one_line(BASELINE)} (mean wall time, higher is better)")
    ax.set_xlim(0, df["speedup_vs_baseline"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("Speedup over the FP32 baseline")
    precision_legend(ax, df["precision"], loc="upper right")
    subtitle(fig, f"Mean of {int(df['wall_s_count'].iloc[0])} runs per binary "
                  f"({len(data.prompts)} prompts × repeats). Wall time in brackets.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ----------------------------------------------------------------------------
# 2. Thread-scaling lines


def plot_thread_scaling(data: RunData):
    df = data.sweep_summary
    if df.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    threads = sorted(df["threads"].unique())
    binaries = [b for b in df["binary"].cat.categories if b in set(df["binary"])]

    ax1.plot(threads, threads, color=NEUTRAL, linestyle="--", linewidth=1.5, label="Ideal (linear)")
    # Labels go on the outer side of each line: above-left for the line that is higher on
    # average, below-right for the others, so a value never sits next to another series' point
    above, below = ((-7, 7), "right", "bottom"), ((7, -7), "left", "top")

    def placements(col):
        """Top line labels above, bottom line below; a middle line goes on the side away from
        its nearest neighbour line."""
        means = df.groupby("binary", observed=True)[col].mean().sort_values(ascending=False)
        names, vals = means.index.tolist(), means.to_numpy()
        out = {}
        for k, b in enumerate(names):
            if k == 0:
                out[b] = above
            elif k == len(names) - 1:
                out[b] = below
            else:
                out[b] = below if vals[k - 1] - vals[k] < vals[k] - vals[k + 1] else above
        return out

    place = {"speedup_vs_1thread": placements("speedup_vs_1thread"), "tok_s_mean": placements("tok_s_mean")}

    for i, b in enumerate(binaries):
        rows = df[df["binary"] == b].sort_values("threads")
        colour = SERIES[i]
        for ax, col, fmt, skip_one in ((ax1, "speedup_vs_1thread", "{:.2f}×", True),
                                       (ax2, "tok_s_mean", "{:.1f}", False)):
            offset, ha, va = place[col][b]
            ax.plot(rows["threads"], rows[col], color=colour, marker="o", label=one_line(b),
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            for x, v in zip(rows["threads"], rows[col]):
                if skip_one and x == 1:
                    continue  # 1.00× by definition
                ax.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=offset,
                            fontsize=8, color=INK, ha=ha, va=va)

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.minorticks_off()
        ax.set_xlabel("OpenMP threads")
    ax1.set_ylabel("Speedup vs 1 thread")
    ax1.set_title("(a) Speedup vs 1 thread")
    ax1.legend(loc="upper left")
    ax2.set_ylabel("Tokens per second (mean)")
    ax2.set_title("(b) Throughput")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, df["tok_s_mean"].max() * 1.15)
    fig.suptitle("OpenMP thread scaling", x=0.01, ha="left", fontweight="bold", fontsize=13)
    subtitle(fig, "Each point is the mean over all prompts and repeats at that thread count (1 thread = 1.00×). "
                  "Efficiency at 8 threads: " + ", ".join(
                      f"{one_line(b)} {df[(df['binary'] == b) & (df['threads'] == max(threads))]['efficiency'].iloc[0]:.0%}"
                      for b in binaries))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ----------------------------------------------------------------------------
# 3. Perplexity vs speed


def plot_perplexity_speed(data: RunData):
    ppl = data.perplexity
    if ppl.empty:
        return None
    speed = data.timing_summary[["binary", "speedup_vs_baseline", "threads"]].copy()
    speed["binary"] = speed["binary"].astype(str)
    df = ppl.assign(binary=ppl["binary"].astype(str)).merge(speed, on="binary", how="inner")
    df = df.sort_values("speedup_vs_baseline").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for p in PRECISION_COLOURS:
        rows = df[df["precision"] == p]
        if rows.empty:
            continue
        ax.scatter(rows["speedup_vs_baseline"], rows["perplexity"], s=90, color=PRECISION_COLOURS[p],
                   edgecolor=SURFACE, linewidth=2, zorder=3, label=p)

    # labels (sorted by speed) step through four heights, alternating above/below the points,
    # so neighbouring points never share a label height
    offsets = [(0, 24), (0, -36), (0, 58), (0, -70)]
    for i, row in df.iterrows():
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(f"{one_line(row['binary'])}\nPPL {row['perplexity']:.6f} · {row['speedup_vs_baseline']:.2f}×",
                    (row["speedup_vs_baseline"], row["perplexity"]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va="center", fontsize=7.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=NEUTRAL, linewidth=0.8))

    ref = df.loc[df["binary"] == BASELINE, "perplexity"]
    if not ref.empty:
        ax.axhline(ref.iloc[0], color=NEUTRAL, linestyle="--", linewidth=1, zorder=1)
        ax.text(0.995, ref.iloc[0], f"FP32 reference {ref.iloc[0]:.6f} ", transform=ax.get_yaxis_transform(),
                color=INK_2, fontsize=8, ha="right", va="top")

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5×", "1×", "2×", "4×", "8×", "16×"])
    ax.minorticks_off()
    ax.set_xlim(df["speedup_vs_baseline"].min() * 0.7, df["speedup_vs_baseline"].max() * 1.5)
    span = df["perplexity"].max() - df["perplexity"].min()
    pad = max(span, 1e-3) * 0.6
    ax.set_ylim(df["perplexity"].min() - pad, df["perplexity"].max() + pad)
    ax.set_xlabel("Speedup vs FP32 baseline (log scale, higher is better)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Accuracy vs speed")
    ax.legend(title="Precision", loc="upper right")
    text = ppl["text"].iloc[0] if "text" in ppl else ""
    subtitle(fig, f"Perplexity on {text} ({int(ppl['tokens'].iloc[0])} tokens, teacher forcing). "
                  "Δ vs FP32: " + ", ".join(
                      f"{p} {df[df['precision'] == p]['ppl_delta_pct'].max():+.4f}%"
                      for p in ("FP16", "INT8") if (df["precision"] == p).any()))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ----------------------------------------------------------------------------
# 4. Box plots of raw tok/s


def boxes(ax, groups, positions, colours, width=0.6):
    """Box (IQR, whiskers 1.5×IQR), every raw run as a dot, mean as a diamond with its value."""
    rng = np.random.default_rng(0)
    bp = ax.boxplot(groups, positions=positions, widths=width, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.5),
                    whiskerprops=dict(color=INK_2), capprops=dict(color=INK_2), boxprops=dict(edgecolor=INK_2))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)
    for pos, values, colour in zip(positions, groups, colours):
        jitter = rng.uniform(-width * 0.3, width * 0.3, len(values))
        ax.scatter(pos + jitter, values, s=8, color=colour, alpha=0.7, linewidth=0, zorder=3)
        mean = float(np.mean(values))
        ax.scatter([pos], [mean], marker="D", s=40, facecolor=SURFACE, edgecolor=INK, linewidth=1.2, zorder=4)
        ax.annotate(f"{mean:.1f}", (pos, np.max(values)), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8, color=INK)


def box_legend_handles(extra):
    return extra + [
        Line2D([], [], marker="D", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK, label="Mean (value above)"),
        Line2D([], [], color=INK, linewidth=1.5, label="Median"),
        Line2D([], [], marker="o", linestyle="none", markersize=4, color=INK_2, label="One run"),
    ]


def plot_tok_s_boxplot(data: RunData):
    timing, sweep = data.timing, data.thread_sweep
    has_sweep = not sweep.empty
    fig, axes = plt.subplots(2 if has_sweep else 1, 1, figsize=(13, 10 if has_sweep else 5.5), squeeze=False)
    ax = axes[0, 0]

    binaries = [b for b in timing["binary"].cat.categories if b in set(timing["binary"])]
    groups = [timing.loc[timing["binary"] == b, "tok_s"].dropna().to_numpy() for b in binaries]
    precisions = [timing.loc[timing["binary"] == b, "precision"].iloc[0] for b in binaries]
    boxes(ax, groups, np.arange(len(binaries)), [PRECISION_COLOURS.get(p, NEUTRAL) for p in precisions])
    ax.set_xticks(np.arange(len(binaries)), [label(b) for b in binaries], fontsize=8.5)
    ax.set_ylabel("Tokens per second")
    ax.set_ylim(0, max(g.max() for g in groups) * 1.12)
    ax.set_title("(a) Timing runs: every prompt × repeat (default thread count)")
    ax.legend(handles=box_legend_handles(
        [Patch(facecolor=c, alpha=0.6, label=p) for p, c in PRECISION_COLOURS.items() if p in precisions]),
        loc="upper left", ncol=2)

    if has_sweep:
        ax = axes[1, 0]
        sb = [b for b in sweep["binary"].cat.categories if b in set(sweep["binary"])]
        threads = sorted(sweep["threads"].unique())
        groups, positions, colours, ticks, tick_pos = [], [], [], [], []
        for i, b in enumerate(sb):
            for j, t in enumerate(threads):
                pos = i * (len(threads) + 1) + j
                groups.append(sweep.loc[(sweep["binary"] == b) & (sweep["threads"] == t), "tok_s"].dropna().to_numpy())
                positions.append(pos)
                colours.append(SERIES[i])
                ticks.append(f"{t}")
                tick_pos.append(pos)
        boxes(ax, groups, np.array(positions, dtype=float), colours, width=0.7)
        ax.set_xticks(tick_pos, [f"{t} thr" for t in ticks])
        for i, b in enumerate(sb):  # binary name under each group of boxes
            centre = i * (len(threads) + 1) + (len(threads) - 1) / 2
            ax.text(centre, -0.1, one_line(b), transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Tokens per second")
        ax.set_ylim(0, max(g.max() for g in groups) * 1.12)
        ax.set_title("(b) Thread sweep: every prompt × repeat per thread count")
        ax.legend(handles=box_legend_handles([Patch(facecolor=SERIES[i], alpha=0.6, label=one_line(b))
                                              for i, b in enumerate(sb)]), loc="upper left", ncol=2)

    fig.suptitle("Distribution of tokens/s over all runs", x=0.01, ha="left", fontweight="bold", fontsize=13)
    subtitle(fig, "tok/s as printed by each binary: Runners all include time forward() and sampling")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ----------------------------------------------------------------------------
# 5. Cachegrind miss rates


def plot_cachegrind(data: RunData):
    df = data.cachegrind
    metrics = [m for m in CACHE_COLOURS if m in df.columns]
    if df.empty or not metrics:
        return None
    names = {"I1_miss_rate": "L1 instruction", "D1_miss_rate": "L1 data", "LL_data_miss_rate": "Last-level data"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    width = 0.8 / len(metrics)
    for k, m in enumerate(metrics):
        pos = x + (k - (len(metrics) - 1) / 2) * width
        ax.bar(pos, df[m], width=width, color=CACHE_COLOURS[m], edgecolor=SURFACE, linewidth=2, label=names[m])
        for p, v in zip(pos, df[m]):
            text = f"{v:.2f}" if v >= 0.01 else (f"{v:.4f}" if v >= 0.0001 else "<0.0001")
            ax.text(p, v, text, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [label(b) + ("\n[1 thr]" if "openmp" in str(b) else "") for b in df["binary"]],
                  fontsize=8.5)
    ax.set_ylabel("Miss rate (%)")
    ax.set_ylim(0, df[metrics].to_numpy().max() * 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("Simulated cache miss rates (cachegrind)")
    ax.legend(title="Cache", loc="upper left")
    subtitle(fig, "Cachegrind with a fixed 32 KiB L1 I/D and 8 MiB last-level cache, 16 tokens, 1 thread "
                  "(OpenMP builds run with OMP_NUM_THREADS=1: valgrind cannot model per-core caches). "
                  "Data miss rate = (read + write misses) / (reads + writes).")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ----------------------------------------------------------------------------
# 6. perf: share of time per function group

PERF_GROUPS = ["matmul", "fp16_to_fp32", "forward (rest of layer)", "OpenMP runtime", "libc / libm"]


def perf_group(function: str) -> str:
    f = function.lower()
    if "fp16_to_fp32" in f:
        return "fp16_to_fp32"
    if "matmul" in f:
        return "matmul"
    if f.startswith("forward"):
        return "forward (rest of layer)"
    if "gomp" in f or "omp" in f:
        return "OpenMP runtime"
    if "glibc" in f or "libc" in f or "libm" in f or f.startswith(("exp", "sqrt", "pow", "mem")):
        return "libc / libm"
    return "Other / unknown"


def plot_perf_breakdown(data: RunData):
    df = data.perf_report
    if df.empty:
        return None
    df = df.assign(group=df["function"].map(perf_group))
    share = df.pivot_table(index="binary", columns="group", values="self_pct", aggfunc="sum",
                           fill_value=0, observed=True)
    share["Other / unknown"] = share.get("Other / unknown", 0) + (100 - share.sum(axis=1)).clip(lower=0)
    groups = [g for g in PERF_GROUPS if g in share.columns and share[g].sum() > 0] + ["Other / unknown"]
    colours = dict(zip(PERF_GROUPS, SERIES), **{"Other / unknown": NEUTRAL})

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(share))[::-1]
    left = np.zeros(len(share))
    for g in groups:
        values = share[g].to_numpy()
        ax.barh(y, values, left=left, color=colours[g], edgecolor=SURFACE, linewidth=2, height=0.7, label=g)
        for yi, l, v in zip(y, left, values):
            if v >= 4:
                ax.text(l + v / 2, yi, f"{v:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if g in ("matmul", "OpenMP runtime") or g == "Other / unknown" else INK)
        left += values
    ax.set_yticks(y, [label(b) for b in share.index], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of CPU samples (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the time goes (perf record, self time)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(groups))
    subtitle(fig, "Functions grouped; labels shown for segments ≥ 4%. In the FP32/FP16 _Float16 baselines matmul "
                  "is attributed to forward(), so it appears as 'forward'.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ----------------------------------------------------------------------------
# 7. Cachegrind thread sweep: total misses per thread count


def thread_colours(n: int) -> list:
    """n ordered, clearly different colours (viridis: green-yellow = few, purple = many threads).
    Sampled from one perceptually uniform, colour-blind-safe map, so any number of thread
    counts gets its own colour and the order still reads light -> dark."""
    cmap = matplotlib.colormaps["viridis"]
    return [cmap(v) for v in np.linspace(0.85, 0.05, n)] if n > 1 else [cmap(0.45)]


def plot_cachegrind_threads(data: RunData):
    df = data.cachegrind_sweep
    if df.empty:
        return None
    panels = [("D1_misses", "(a) L1 data misses"), ("LL_data_misses", "(b) Last-level data misses"),
              ("Ir", "(c) Instructions executed")]
    threads = sorted(df["threads"].unique())
    binaries = [b for b in df["binary"].cat.categories if b in set(df["binary"])]
    colours = thread_colours(len(threads))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    x = np.arange(len(binaries))
    width = 0.8 / len(threads)
    for ax, (col, title) in zip(axes, panels):
        top = df[col].max() / 1e6
        for k, t in enumerate(threads):
            rows = df[df["threads"] == t].set_index(df[df["threads"] == t]["binary"].astype(str))
            values = np.array([rows[col].get(b, np.nan) for b in binaries], dtype=float) / 1e6
            ratios = [rows[f"{col}_vs_1thread"].get(b, np.nan) for b in binaries]
            pos = x + (k - (len(threads) - 1) / 2) * width
            ax.bar(pos, values, width=width, color=colours[k], edgecolor=SURFACE, linewidth=1.5,
                   label=f"{t} thread{'s' if t > 1 else ''}")
            for p_, v, r in zip(pos, values, ratios):  # value, and ratio vs 1 thread for t > 1
                if np.isnan(v):
                    continue
                # change vs 1 thread as a % (the ratios are ~1.00x, so 2 decimals of x hide it)
                text = f"{v:,.1f}M" if t == threads[0] else f"{v:,.1f}M  {100 * (r - 1):+.1f}%"
                ax.text(p_, v + top * 0.01, text, rotation=90, ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x, [label(b) for b in binaries], fontsize=8.5)
        ax.set_ylim(0, top * 1.45)  # room for the rotated labels
        ax.set_ylabel("Millions (sum over all threads)")
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, title="OpenMP threads", loc="upper right", ncol=len(threads),
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Total simulated cache misses vs thread count (cachegrind)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    steps = data.meta["valgrind_steps"].iloc[0] if "valgrind_steps" in data.meta else float("nan")
    subtitle(fig, f"Sum over all threads, {steps:.0f} tokens, OMP_WAIT_POLICY=passive. Valgrind runs threads one at "
                  "a time on ONE simulated 32 KiB L1 / 8 MiB LL cache, so totals are comparable, per-core "
                  "behaviour is not. % = change vs the same build at 1 thread.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return fig


# ----------------------------------------------------------------------------
# 8. Model file size and peak memory per precision


def plot_model_size(data: RunData):
    meta = data.meta
    if "model_bytes" not in meta or meta["model_bytes"].isna().all():
        return None
    has_rss = not data.time_v.empty and "max_rss_kb" in data.time_v
    fig, axes = plt.subplots(1, 2 if has_rss else 1, figsize=(14 if has_rss else 7, 5.5), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.6]} if has_rss else None)

    # (a) one bar per model file (FP32 / FP16 / INT8), not per binary
    files = (meta.assign(binary=meta["binary"].astype(str))
             .dropna(subset=["model_bytes"])
             .drop_duplicates("model")[["model", "precision", "model_bytes"]])
    order = list(PRECISION_COLOURS)
    files = files.sort_values("precision", key=lambda c: c.map(order.index)).reset_index(drop=True)
    fp32 = files.loc[files["precision"] == "FP32", "model_bytes"]
    ax = axes[0, 0]
    x = np.arange(len(files))
    mb = files["model_bytes"] / 1e6
    ax.bar(x, mb, color=[PRECISION_COLOURS.get(p, NEUTRAL) for p in files["precision"]], width=0.6,
           edgecolor=SURFACE, linewidth=2)
    for xi, (_, row) in zip(x, files.iterrows()):
        note = "" if fp32.empty or row["precision"] == "FP32" else \
            f"\n{fp32.iloc[0] / row['model_bytes']:.2f}× smaller"
        ax.text(xi, row["model_bytes"] / 1e6, f"{row['model_bytes'] / 1e6:.1f} MB{note}", ha="center",
                va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{p}\n{m}" for p, m in zip(files["precision"], files["model"])], fontsize=8.5)
    ax.set_ylim(0, mb.max() * 1.2)
    ax.set_ylabel("Model file size (MB)")
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Model file (.bin) size")
    precision_legend(ax, files["precision"], loc="upper right")

    # (b) peak resident memory of every binary while generating
    if has_rss:
        rss = data.time_v.dropna(subset=["max_rss_kb"])
        ax = axes[0, 1]
        y = np.arange(len(rss))[::-1]
        values = rss["max_rss_kb"] * 1024 / 1e6  # time -v reports KiB; MB = 10^6 bytes as in (a)
        ax.barh(y, values, color=[PRECISION_COLOURS.get(p, NEUTRAL) for p in rss["precision"]], height=0.7,
                edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, values):
            ax.text(v, yi, f"  {v:.0f} MB", va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_yticks(y, [one_line(b) for b in rss["binary"]], fontsize=8.5)
        ax.set_xlim(0, values.max() * 1.18)
        ax.set_xlabel("Peak resident memory, max RSS (MB)")
        ax.grid(axis="y", visible=False)
        ax.set_title("(b) Peak memory while generating")

    fig.suptitle("Model size by precision", x=0.01, ha="left", fontweight="bold", fontsize=13)
    subtitle(fig, "MB = 10^6 bytes. Max RSS from /usr/bin/time -v (first prompt, default threads); it includes the "
                  "mmap-ed weights, the KV cache and buffers, and the INT8 build's FP32 copy of the token embeddings.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ----------------------------------------------------------------------------
# 9. Next-token agreement vs FP32

MISMATCH_KINDS = [  # column, legend label, colour
    ("ref_right_only", "FP32 right, this binary wrong", SERIES[3]),
    ("cand_right_only", "This binary right, FP32 wrong", SERIES[4]),
    ("both_wrong", "Both wrong (different tokens)", NEUTRAL),
]


def plot_agreement(data: RunData):
    df = data.agreement
    if df.empty:
        return None
    ref = str(df["reference"].iloc[0])
    has_kinds = all(c in df.columns for c, _, _ in MISMATCH_KINDS) and df["ref_right_only"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_kinds else 1, figsize=(14 if has_kinds else 9, 5.5),
                             squeeze=False, sharey=True)
    y = np.arange(len(df))[::-1]  # first binary at the top

    # (a) flip rate: share of positions whose argmax differs from the reference
    ax = axes[0, 0]
    colours = [PRECISION_COLOURS.get(p, NEUTRAL) for p in df["precision"]]
    ax.barh(y, df["flip_pct"], color=colours, height=0.7, edgecolor=SURFACE, linewidth=2)
    for yi, (_, row) in zip(y, df.iterrows()):
        note = "reference" if str(row["binary"]) == ref else \
            f"{int(row['mismatches'])} of {int(row['positions'])} differ"
        ax.text(row["flip_pct"], yi, f"  {row['agreement_pct']:.2f}% agree · {note}",
                va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y, [label(b) for b in df["binary"]], fontsize=8.5)
    ax.set_xlim(0, max(df["flip_pct"].max(), 0.5) * 1.9)  # room for the labels; 0.5% if nothing differs
    ax.set_xlabel(f"Positions whose top-1 token differs from {one_line(ref)} (%, lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) Disagreement with FP32 (1 − agreement)")
    precision_legend(ax, df["precision"], loc="lower right")

    # (b) what kind of mismatch: did the change lose or gain a correct prediction?
    if has_kinds:
        ax = axes[0, 1]
        left = np.zeros(len(df))
        for col, name, colour in MISMATCH_KINDS:
            values = df[col].fillna(0).to_numpy(dtype=float)
            ax.barh(y, values, left=left, color=colour, height=0.7, edgecolor=SURFACE, linewidth=2, label=name)
            for yi, l, v in zip(y, left, values):
                if v > 0:
                    ax.text(l + v / 2, yi, f"{int(v)}", ha="center", va="center", fontsize=8,
                            color="white" if colour == NEUTRAL else INK)
            left += values
        for yi, total, (_, row) in zip(y, left, df.iterrows()):
            acc = row.get("top1_acc_pct", np.nan)
            ax.text(total, yi, f"  top-1 acc {acc:.2f}%" if pd.notna(acc) else "", va="center",
                    ha="left", fontsize=8.5, color=INK_2)
        ax.set_xlim(0, max(left.max(), 1) * 1.6)
        ax.set_xlabel("Mismatched positions (count)")
        ax.grid(axis="y", visible=False)
        ax.set_title("(b) Kind of mismatch, and top-1 accuracy on the true text")
        ax.legend(loc="lower right")

    fig.suptitle("Next-token agreement with the FP32 baseline", x=0.01, ha="left", fontweight="bold", fontsize=13)
    text = df["text"].iloc[0] if "text" in df else ""
    subtitle(fig, f"Teacher forcing on {text} ({int(df['positions'].iloc[0])} positions): at each position "
                  "both models see the same true context; argmax of the logits compared (temperature-independent). "
                  "Top-1 accuracy = argmax equals the true next token.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# ----------------------------------------------------------------------------


FIGURES = [
    ("1_speedup", plot_speedup),
    ("2_thread_scaling", plot_thread_scaling),
    ("3_perplexity_speed", plot_perplexity_speed),
    ("4_tok_s_boxplot", plot_tok_s_boxplot),
    ("5_cachegrind", plot_cachegrind),
    ("6_perf_breakdown", plot_perf_breakdown),
    ("7_cachegrind_threads", plot_cachegrind_threads),
    ("8_model_size", plot_model_size),
    ("9_next_token_agreement", plot_agreement),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", type=Path, help="logs/profile/<run_id> (default: latest)")
    parser.add_argument("--show", action="store_true", help="open the figures after saving")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    setup_style()
    run_dir = (args.run_dir if args.run_dir else latest_run_dir()).resolve()
    data = load_run(run_dir)
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)

    for name, make in FIGURES:
        fig = make(data)
        if fig is None:
            print(f"skipped {name} (no data)")
            continue
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        shown = out / name
        print(f"wrote {shown.relative_to(ROOT) if shown.is_relative_to(ROOT) else shown}.png/.pdf")
        if not args.show:
            plt.close(fig)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
