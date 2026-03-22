#!/usr/bin/env python3
"""
Multi-Dataset λ-Frontier Benchmark Runner.

Runs λ-frontier sweeps across multiple datasets and produces:
1. Per-dataset CSV files and plots
2. Cross-dataset summary table
3. Method ranking analysis

Usage:
    # Run all datasets (5 splits each)
    python scripts/run_benchmark.py

    # Run specific datasets
    python scripts/run_benchmark.py --datasets adult,bank,spambase

    # Quick test run (2 splits)
    python scripts/run_benchmark.py --n-splits 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import list_datasets
from lambda_frontier import (
    LAMBDAS_TRAIN,
    METHOD_COLORS,
    METHOD_MARKERS,
    run_dataset_sweep,
    summarize,
)


def aggregate_results(outdir: Path, datasets: List[str]) -> pd.DataFrame:
    """Combine per-dataset results into a single dataframe."""
    dfs = []
    for ds in datasets:
        path = outdir / f"{ds}_lambda_frontier_raw.csv"
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def compute_cross_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics across datasets for each method."""
    rows = []

    for method in df["method"].unique():
        mdf = df[df["method"] == method]

        for lam in mdf["lam"].unique():
            ldf = mdf[mdf["lam"] == lam]

            summary_per_ds = ldf.groupby("dataset").agg(
                acc=("acc", "mean"),
                nfr=("nfr", "mean"),
                cum_forgetting=("cum_forgetting", "mean"),
            )

            rows.append({
                "method": method,
                "lam": lam,
                "n_datasets": len(summary_per_ds),
                "acc_mean": summary_per_ds["acc"].mean(),
                "acc_std": summary_per_ds["acc"].std(),
                "nfr_mean": summary_per_ds["nfr"].mean(),
                "nfr_std": summary_per_ds["nfr"].std(),
                "cum_forgetting_mean": summary_per_ds["cum_forgetting"].mean(),
                "cum_forgetting_std": summary_per_ds["cum_forgetting"].std(),
            })

    return pd.DataFrame(rows)


def compute_method_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute method rankings per dataset.

    For each dataset, rank methods by their best (minimum) NFR achieved
    while maintaining accuracy within 1% of baseline.
    """
    rows = []

    for dataset in df["dataset"].unique():
        ddf = df[df["dataset"] == dataset]

        baseline_acc = ddf[ddf["method"] == "baseline"]["acc"].mean()
        acc_threshold = baseline_acc - 0.01

        method_best_nfr = {}
        for method in ddf["method"].unique():
            mdf = ddf[ddf["method"] == method]

            valid = mdf.groupby("lam").agg(
                acc=("acc", "mean"),
                nfr=("nfr", "mean"),
            )
            valid = valid[valid["acc"] >= acc_threshold]

            if len(valid) > 0:
                method_best_nfr[method] = valid["nfr"].min()
            else:
                method_best_nfr[method] = float("inf")

        sorted_methods = sorted(method_best_nfr.items(), key=lambda x: x[1])
        for rank, (method, best_nfr) in enumerate(sorted_methods, 1):
            rows.append({
                "dataset": dataset,
                "method": method,
                "rank": rank,
                "best_nfr": best_nfr if best_nfr != float("inf") else np.nan,
                "baseline_acc": baseline_acc,
            })

    return pd.DataFrame(rows)


def plot_cross_dataset_frontiers(df: pd.DataFrame, figdir: Path) -> None:
    """Plot combined Pareto frontiers across all datasets."""
    summary = df.groupby(["method", "lam"]).agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        nfr_mean=("nfr", "mean"),
        nfr_std=("nfr", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))

    for method in summary["method"].unique():
        mdf = summary[summary["method"] == method].copy()
        mdf = mdf.sort_values("lam")

        color = METHOD_COLORS.get(method, "gray")
        marker = METHOD_MARKERS.get(method, "o")

        x = mdf["nfr_mean"].values
        y = 1.0 - mdf["acc_mean"].values
        xerr = mdf["nfr_std"].values
        yerr = mdf["acc_std"].values

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            color=color,
            markersize=8,
            label=method,
            capsize=3,
            alpha=0.8,
        )

    ax.set_xlabel("Negative Flip Rate (NFR)", fontsize=12)
    ax.set_ylabel("Error Rate (1 - Accuracy)", fontsize=12)
    ax.set_title("Cross-Dataset Pareto Frontiers", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "cross_dataset_pareto.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(figdir / "cross_dataset_pareto.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {figdir / 'cross_dataset_pareto.pdf'}")


def plot_method_rankings(ranks_df: pd.DataFrame, figdir: Path) -> None:
    """Plot method rankings across datasets."""
    methods = ranks_df["method"].unique()
    datasets = ranks_df["dataset"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.15
    offsets = np.linspace(-width * (len(methods) - 1) / 2, width * (len(methods) - 1) / 2, len(methods))

    for i, method in enumerate(methods):
        mdf = ranks_df[ranks_df["method"] == method]
        ranks = [mdf[mdf["dataset"] == ds]["rank"].values[0] if len(mdf[mdf["dataset"] == ds]) > 0 else np.nan
                 for ds in datasets]

        color = METHOD_COLORS.get(method, "gray")
        ax.bar(x + offsets[i], ranks, width, label=method, color=color, alpha=0.8)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Rank (lower is better)", fontsize=12)
    ax.set_title("Method Rankings Across Datasets", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.5, len(methods) + 0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(figdir / "method_rankings.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(figdir / "method_rankings.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {figdir / 'method_rankings.pdf'}")


def print_summary_table(ranks_df: pd.DataFrame) -> None:
    """Print a summary table of method performance."""
    print("\n" + "=" * 70)
    print("CROSS-DATASET METHOD SUMMARY")
    print("=" * 70)

    avg_rank = ranks_df.groupby("method")["rank"].mean().sort_values()
    win_count = ranks_df[ranks_df["rank"] == 1].groupby("method").size()

    print(f"\n{'Method':<20} {'Avg Rank':>10} {'Wins':>8}")
    print("-" * 40)
    for method in avg_rank.index:
        wins = win_count.get(method, 0)
        print(f"{method:<20} {avg_rank[method]:>10.2f} {wins:>8}")

    print("\n" + "-" * 70)
    print("Per-dataset rankings (lower = better):")
    print("-" * 70)

    pivot = ranks_df.pivot(index="method", columns="dataset", values="rank")
    print(pivot.to_string())


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset λ-Frontier Benchmark")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"Comma-separated list of datasets to run. "
        f"Default: all ({', '.join(list_datasets())})",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("tabs"),
        help="Output directory for CSV files (default: tabs/)",
    )
    parser.add_argument(
        "--figdir",
        type=Path,
        default=Path("figs"),
        help="Output directory for figures (default: figs/)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of random splits for statistical reliability (default: 5)",
    )
    args = parser.parse_args()

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
        for d in datasets:
            if d not in list_datasets():
                raise ValueError(f"Unknown dataset: {d}. Available: {list_datasets()}")
    else:
        datasets = list_datasets()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    print(f"Running benchmark on {len(datasets)} datasets: {', '.join(datasets)}")
    print(f"Splits per dataset: {args.n_splits}")

    for ds in datasets:
        run_dataset_sweep(
            dataset_name=ds,
            n_splits=args.n_splits,
            outdir=args.outdir,
            figdir=args.figdir,
        )

    print("\n" + "=" * 70)
    print("AGGREGATING CROSS-DATASET RESULTS")
    print("=" * 70)

    combined_df = aggregate_results(args.outdir, datasets)
    if len(combined_df) == 0:
        print("No results to aggregate!")
        return

    combined_path = args.outdir / "all_datasets_raw.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved: {combined_path}")

    cross_summary = compute_cross_dataset_summary(combined_df)
    cross_summary_path = args.outdir / "cross_dataset_summary.csv"
    cross_summary.to_csv(cross_summary_path, index=False)
    print(f"Saved: {cross_summary_path}")

    ranks_df = compute_method_ranks(combined_df)
    ranks_path = args.outdir / "method_rankings.csv"
    ranks_df.to_csv(ranks_path, index=False)
    print(f"Saved: {ranks_path}")

    print("\nGenerating cross-dataset plots...")
    plot_cross_dataset_frontiers(combined_df, args.figdir)
    plot_method_rankings(ranks_df, args.figdir)

    print_summary_table(ranks_df)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
