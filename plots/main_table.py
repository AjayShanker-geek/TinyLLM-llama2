"""
main_table.py - the main results table of the report, from one profile run.

One row per binary: median tok/s and its IQR over all prompts and repeats, speedup over
run_fp32_baseline (ratio of median tok/s), weight file size, estimated bandwidth, perplexity
and next-token agreement with the baseline. Loading is shared with plots.py (data_summary.py).

Usage (from the project root):
    .venv/bin/python plots/main_table.py                                # latest run in logs/profile
    .venv/bin/python plots/main_table.py logs/profile/20261107-164852   # a specific run

Output: the table on stdout and in <run_dir>/tables/main_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_summary import BASELINE, ROOT, RunData, latest_run_dir, load_run


def main_results_table(data: RunData) -> pd.DataFrame:
    """Returns a DataFrame with one row per binary, in the same order as the figures."""
    per_binary = data.timing.groupby("binary", observed=True)
    tok_s = per_binary["tok_s"]
    table = pd.DataFrame(
        {
            "tok_s_median": tok_s.median(),
            "tok_s_iqr": tok_s.quantile(0.75) - tok_s.quantile(0.25),
            "bandwidth_gb_s": per_binary["bandwidth"].median(),
            "threads": per_binary["threads"].first().astype(int),
            "runs": tok_s.count(),
            "variant": per_binary["precision"].first() + " " + per_binary["method"].first(),
        }
    )
    table["speedup"] = table["tok_s_median"] / table.loc[BASELINE, "tok_s_median"]

    table = table.join(data.meta.set_index("binary")["model_bytes"])
    table["weights_mb"] = table["model_bytes"] / 1e6
    table = table.join(data.perplexity.set_index("binary")["perplexity"])
    table = table.join(data.agreement.set_index("binary")["agreement_pct"])

    columns = [
        "variant",
        "threads",
        "runs",
        "tok_s_median",
        "tok_s_iqr",
        "speedup",
        "weights_mb",
        "bandwidth_gb_s",
        "perplexity",
        "agreement_pct",
    ]
    return table[columns].reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir", nargs="?", type=Path, help="logs/profile/<run_id> (default: latest)"
    )
    args = parser.parse_args()

    run_dir = (args.run_dir if args.run_dir else latest_run_dir()).resolve()
    table = main_results_table(load_run(run_dir))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(table.drop(columns="binary").round(
        {"tok_s_median": 2, "tok_s_iqr": 2, "speedup": 2, "weights_mb": 0, "bandwidth_gb_s": 1,
         "perplexity": 6, "agreement_pct": 2}).to_string(index=False))

    out = run_dir / "tables" / "main_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwritten to {out.relative_to(ROOT)}")
