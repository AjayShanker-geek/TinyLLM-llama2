"""
plots.py - load the CSVs written by scripts/profile.sh into tidy pandas tables.

Usage (from the project root):
    .venv/bin/python plots/plots.py                                  # latest run in logs/profile
    .venv/bin/python plots/plots.py logs/profile/20261104-105307     # a specific run
    .venv/bin/python plots/plots.py logs/profile/20261104-105307 --save   # also write tables to CSV

Input, in the run folder (see the header of scripts/profile.sh for the columns):
    results.csv     long format: one row per measurement (tool, metric, item, threads, repeat, value)
    perplexity.csv  one row per binary
    next_token_agreement.csv  one row per binary: matches vs the FP32 baseline's argmax
    <binary>/next_token_trace.csv  teacher-forced argmax per position (for top-1 accuracy)
    prompts.tsv     prompt id (p01..) -> prompt text

Every table below has one row per observation and named columns, ready for plotting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: F401  (used by the plotting step, added next)
import numpy as np  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / "logs" / "profile"

BASELINE = "run_fp32_baseline"  # speedups are relative to this binary

# Display order and grouping for plots; binaries not listed here are appended at the end
BINARY_INFO = {
    #  binary                       precision  method
    "run_fp32_baseline":          ("FP32", "baseline"),
    "run_fp16_baseline":          ("FP16", "baseline (software decode)"),
    "run_fp16_baseline_native":   ("FP16", "baseline (_Float16)"),
    "run_int8_baseline":          ("INT8", "baseline"),
    "run_fp32_simd":              ("FP32", "SIMD"),
    "run_fp32_simd_4row":         ("FP32", "SIMD 4-row"),
    "run_fp32_openmp":            ("FP32", "OpenMP"),
    "run_fp32_openmp_simd":       ("FP32", "OpenMP + -O3 SIMD"),
    "run_fp32_openmp_neon":       ("FP32", "OpenMP + NEON"),
}


# ----------------------------------------------------------------------------
# Loading


@dataclass
class RunData:
    run_dir: Path
    raw: pd.DataFrame               # results.csv as is
    meta: pd.DataFrame              # binary x setting (steps, runs, seed, prompts, model_bytes, ...)
    prompts: pd.DataFrame           # prompt_id, prompt
    timing: pd.DataFrame            # binary, prompt_id, repeat, threads, wall_s, tok_s, bandwidth
    timing_summary: pd.DataFrame    # per binary: mean/std/... of wall_s, tok_s, bandwidth + speedup
    thread_sweep: pd.DataFrame      # like timing, for each thread count (OpenMP builds)
    sweep_summary: pd.DataFrame     # per binary x threads: means + speedup and efficiency vs 1 thread
    time_v: pd.DataFrame            # /usr/bin/time -v: binary x metric (max_rss_kb, cpu_percent, ...)
    perf_stat: pd.DataFrame         # binary x metric (task_clock_ms, context_switches, ...)
    perf_report: pd.DataFrame       # binary, function, self_pct
    gprof: pd.DataFrame             # binary, function, self_pct, self_s, calls
    callgrind: pd.DataFrame         # binary, function, Ir, pct_of_total
    callgrind_total: pd.DataFrame   # binary, Ir_total
    cachegrind: pd.DataFrame        # binary x event totals and miss rates
    cachegrind_functions: pd.DataFrame  # binary, function, D1_misses
    massif: pd.DataFrame            # binary, peak_heap_bytes, peak_heap_extra_bytes
    perplexity: pd.DataFrame        # perplexity.csv + precision/method + delta vs FP32
    agreement: pd.DataFrame         # next_token_agreement.csv + flip rate, top-1 accuracy, mismatch kinds


def latest_run_dir() -> Path:
    runs = sorted(p for p in PROFILE_DIR.iterdir() if (p / "results.csv").is_file())
    if not runs:
        raise SystemExit(f"no runs with results.csv in {PROFILE_DIR}")
    return runs[-1]


def binary_order(binaries) -> list[str]:
    known = [b for b in BINARY_INFO if b in set(binaries)]
    return known + sorted(set(binaries) - set(known))


def add_binary_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add precision/method columns and make `binary` an ordered categorical (plots keep this order)."""
    df = df.copy()
    df["precision"] = df["binary"].map(lambda b: BINARY_INFO.get(b, ("?", "?"))[0])
    df["method"] = df["binary"].map(lambda b: BINARY_INFO.get(b, ("?", b))[1])
    order = binary_order(df["binary"].unique())
    df["binary"] = pd.Categorical(df["binary"], categories=order, ordered=True)
    return df.sort_values("binary", kind="stable").reset_index(drop=True)


def select(raw: pd.DataFrame, tool: str, metric: str | None = None) -> pd.DataFrame:
    rows = raw[raw["tool"] == tool]
    return rows if metric is None else rows[rows["metric"] == metric]


def wide_per_binary(raw: pd.DataFrame, tool: str) -> pd.DataFrame:
    """One row per binary, one column per metric (for tools that report single totals)."""
    rows = select(raw, tool)
    if rows.empty:
        return pd.DataFrame(columns=["binary"])
    wide = rows.pivot_table(index="binary", columns="metric", values="value", aggfunc="first", observed=True)
    wide.columns.name = None
    return add_binary_info(wide.reset_index())


def per_function(raw: pd.DataFrame, tool: str, metrics: list[str]) -> pd.DataFrame:
    """One row per (binary, function) with a column per metric."""
    rows = raw[(raw["tool"] == tool) & raw["metric"].isin(metrics)]
    if rows.empty:
        return pd.DataFrame(columns=["binary", "function", *metrics])
    wide = rows.pivot_table(index=["binary", "item"], columns="metric", values="value",
                            aggfunc="sum", observed=True)
    wide.columns.name = None
    wide = wide.reset_index().rename(columns={"item": "function"})
    return add_binary_info(wide)


def timing_table(raw: pd.DataFrame, tool: str) -> pd.DataFrame:
    """wall_s / tok_s / bandwidth side by side, one row per (binary, threads, prompt, repeat)."""
    rows = select(raw, tool)
    if rows.empty:
        return pd.DataFrame(columns=["binary", "threads", "prompt_id", "repeat", "wall_s", "tok_s", "bandwidth"])
    wide = rows.pivot_table(index=["binary", "threads", "item", "repeat"], columns="metric",
                            values="value", aggfunc="first", observed=True)
    wide.columns.name = None
    wide = wide.reset_index().rename(columns={"item": "prompt_id"})
    wide["repeat"] = wide["repeat"].astype(int)
    return add_binary_info(wide)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """mean, std, min, median, max and count of the timing metrics, over all prompts and repeats."""
    metrics = [m for m in ("wall_s", "tok_s", "bandwidth") if m in df.columns]
    out = df.groupby(by, observed=True)[metrics].agg(["mean", "std", "min", "median", "max", "count"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index()


def load_trace(path: Path) -> pd.DataFrame:
    """<binary>/next_token_trace.csv from agreement.c (header lines start with #)."""
    return pd.read_csv(path, comment="#")


def agreement_table(run_dir: Path) -> pd.DataFrame:
    """next_token_agreement.csv plus statistics computed from the per-position traces:
    flip_pct           100 - agreement_pct (share of positions whose argmax differs from FP32)
    top1_acc_pct       share of positions where the argmax equals the true next token
    ref_top1_acc_pct   the same for the reference (FP32) trace
    of the mismatches: ref_right_only (FP32 was right, this binary wrong), cand_right_only
    (this binary right, FP32 wrong), both_wrong (both wrong, different tokens)
    """
    path = run_dir / "next_token_agreement.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, keep_default_na=False)
    if df.empty:
        return df
    df["agreement_pct"] = pd.to_numeric(df["agreement_pct"])
    df["flip_pct"] = 100 - df["agreement_pct"]
    if df["text_sha256"].nunique() > 1:
        print("WARNING: agreement rows used different texts (text_sha256 differs)")

    traces = {b: run_dir / b / "next_token_trace.csv" for b in set(df["binary"]) | set(df["reference"])}
    traces = {b: load_trace(p) for b, p in traces.items() if p.is_file()}
    extra = []
    for _, row in df.iterrows():
        cand, ref = traces.get(row["binary"]), traces.get(row["reference"])
        stats = {"binary": row["binary"]}
        if cand is not None:
            stats["top1_acc_pct"] = 100 * (cand["predicted_token"] == cand["target_token"]).mean()
        if cand is not None and ref is not None and len(cand) == len(ref):
            c_ok = cand["predicted_token"].to_numpy() == cand["target_token"].to_numpy()
            r_ok = ref["predicted_token"].to_numpy() == ref["target_token"].to_numpy()
            differ = cand["predicted_token"].to_numpy() != ref["predicted_token"].to_numpy()
            stats["ref_top1_acc_pct"] = 100 * r_ok.mean()
            stats["ref_right_only"] = int((differ & r_ok).sum())
            stats["cand_right_only"] = int((differ & c_ok).sum())
            stats["both_wrong"] = int((differ & ~r_ok & ~c_ok).sum())
            # first position (0-based) where the argmax differs; -1 if never
            stats["first_mismatch"] = int(np.argmax(differ)) if differ.any() else -1
        extra.append(stats)
    df = df.merge(pd.DataFrame(extra), on="binary", how="left")
    return add_binary_info(df)


def load_run(run_dir: Path) -> RunData:
    raw = pd.read_csv(run_dir / "results.csv", keep_default_na=False, na_values=[""])
    raw["item"] = raw["item"].fillna("")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    # meta: settings per binary
    meta = select(raw, "meta").pivot_table(index="binary", columns="metric", values="value",
                                           aggfunc="first").reset_index()
    meta.columns.name = None
    model_names = select(raw, "meta", "model_bytes").set_index("binary")["item"]
    meta["model"] = meta["binary"].map(model_names)
    meta = add_binary_info(meta)

    prompts_file = run_dir / "prompts.tsv"
    prompts = (pd.read_csv(prompts_file, sep="\t", header=None, names=["prompt_id", "prompt"],
                           quoting=3, keep_default_na=False)
               if prompts_file.is_file() else pd.DataFrame(columns=["prompt_id", "prompt"]))

    # timing
    timing = timing_table(raw, "timing")
    timing_summary = add_binary_info(summarise(timing, ["binary", "threads"]))
    base = timing_summary.loc[timing_summary["binary"] == BASELINE, "wall_s_mean"]
    timing_summary["speedup_vs_baseline"] = (base.iloc[0] / timing_summary["wall_s_mean"]
                                             if not base.empty else np.nan)

    # thread sweep
    thread_sweep = timing_table(raw, "thread_sweep")
    sweep_summary = summarise(thread_sweep, ["binary", "threads"]) if not thread_sweep.empty else thread_sweep
    if not sweep_summary.empty:
        one = sweep_summary[sweep_summary["threads"] == 1].set_index("binary")["wall_s_mean"]
        one.index = one.index.astype(str)
        base_1t = sweep_summary["binary"].astype(str).map(one)
        sweep_summary["speedup_vs_1thread"] = base_1t / sweep_summary["wall_s_mean"]
        sweep_summary["efficiency"] = sweep_summary["speedup_vs_1thread"] / sweep_summary["threads"]
        sweep_summary = add_binary_info(sweep_summary)

    # single-run profilers
    time_v = wide_per_binary(raw, "time")
    perf_stat = wide_per_binary(raw, "perf_stat")
    perf_report = per_function(raw, "perf_report", ["self_pct"])
    gprof = per_function(raw, "gprof", ["self_pct", "self_s", "calls"])

    callgrind_total = wide_per_binary(raw, "callgrind")[["binary", "precision", "method", "Ir_total"]] \
        if not select(raw, "callgrind").empty else pd.DataFrame(columns=["binary", "Ir_total"])
    callgrind = per_function(raw, "callgrind", ["Ir"])
    if not callgrind.empty:
        totals = callgrind_total.set_index(callgrind_total["binary"].astype(str))["Ir_total"]
        callgrind["pct_of_total"] = 100 * callgrind["Ir"] / callgrind["binary"].astype(str).map(totals)

    cache_rows = select(raw, "cachegrind")
    cachegrind = wide_per_binary(cache_rows[cache_rows["item"] == ""], "cachegrind")
    cachegrind_functions = per_function(raw, "cachegrind", ["D1_misses"])
    massif = wide_per_binary(raw, "massif")

    # perplexity
    ppl_file = run_dir / "perplexity.csv"
    perplexity = pd.read_csv(ppl_file, keep_default_na=False) if ppl_file.is_file() else pd.DataFrame()
    if not perplexity.empty:
        perplexity = add_binary_info(perplexity)
        ref = perplexity.loc[perplexity["binary"] == BASELINE, "perplexity"]
        if not ref.empty:
            perplexity["ppl_delta_vs_fp32"] = perplexity["perplexity"] - ref.iloc[0]
            perplexity["ppl_delta_pct"] = 100 * perplexity["ppl_delta_vs_fp32"] / ref.iloc[0]
        if perplexity["text_sha256"].nunique() > 1:
            print("WARNING: perplexity rows scored different texts (text_sha256 differs)")

    # next-token agreement (per-binary CSV; results.csv has the same numbers as tool=agreement)
    agreement = agreement_table(run_dir)

    return RunData(run_dir, raw, meta, prompts, timing, timing_summary, thread_sweep, sweep_summary,
                   time_v, perf_stat, perf_report, gprof, callgrind, callgrind_total, cachegrind,
                   cachegrind_functions, massif, perplexity, agreement)


# ----------------------------------------------------------------------------
# Reporting


def print_overview(data: RunData) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:,.4g}")

    print(f"run: {data.run_dir.name}   rows in results.csv: {len(data.raw)}")
    print(f"binaries: {', '.join(data.meta['binary'].astype(str))}")
    print(f"prompts: {len(data.prompts)}\n")

    print("== timing (mean over all prompts and repeats) ==")
    print(data.timing_summary[["binary", "threads", "wall_s_mean", "wall_s_std", "tok_s_mean",
                               "bandwidth_mean", "wall_s_count", "speedup_vs_baseline"]].to_string(index=False))

    if not data.sweep_summary.empty:
        print("\n== thread sweep ==")
        print(data.sweep_summary[["binary", "threads", "wall_s_mean", "tok_s_mean",
                                  "speedup_vs_1thread", "efficiency"]].to_string(index=False))

    if not data.perplexity.empty:
        cols = [c for c in ("binary", "model", "tokens", "mean_nll", "perplexity", "ppl_delta_pct", "wall_s")
                if c in data.perplexity.columns]
        print("\n== perplexity ==")
        print(data.perplexity[cols].to_string(index=False))

    if not data.agreement.empty:
        cols = [c for c in ("binary", "positions", "matches", "mismatches", "agreement_pct", "flip_pct",
                            "top1_acc_pct", "ref_right_only", "cand_right_only", "both_wrong", "first_mismatch")
                if c in data.agreement.columns]
        print(f"\n== next-token agreement vs {data.agreement['reference'].iloc[0]} ==")
        print(data.agreement[cols].to_string(index=False))

    if not data.time_v.empty:
        print("\n== time -v ==")
        print(data.time_v[["binary", "max_rss_kb", "cpu_percent", "user_s", "sys_s"]].to_string(index=False))

    if not data.callgrind_total.empty:
        print("\n== callgrind total instructions ==")
        print(data.callgrind_total[["binary", "Ir_total"]].to_string(index=False))

    if not data.cachegrind.empty:
        cols = [c for c in ("binary", "D1_miss_rate", "LL_data_miss_rate", "I1_miss_rate")
                if c in data.cachegrind.columns]
        print("\n== cachegrind miss rates (%) ==")
        print(data.cachegrind[cols].to_string(index=False))

    if not data.perf_report.empty:
        top = (data.perf_report.sort_values(["binary", "self_pct"], ascending=[True, False])
               .groupby("binary", observed=True).head(3))
        print("\n== perf record: top 3 functions per binary ==")
        print(top[["binary", "function", "self_pct"]].to_string(index=False))


def save_tables(data: RunData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in fields(data):
        table = getattr(data, f.name)
        if isinstance(table, pd.DataFrame) and f.name != "raw" and not table.empty:
            table.to_csv(out_dir / f"{f.name}.csv", index=False)
    print(f"\ntables written to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", type=Path, help="logs/profile/<run_id> (default: latest)")
    parser.add_argument("--save", action="store_true", help="write every table to <run_dir>/tables/*.csv")
    args = parser.parse_args()

    run_dir = (args.run_dir if args.run_dir else latest_run_dir()).resolve()
    data = load_run(run_dir)
    print_overview(data)
    if args.save:
        save_tables(data, run_dir / "tables")
