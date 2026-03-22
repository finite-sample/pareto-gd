#!/usr/bin/env python3
"""
Analysis script for NFR experiment results.

Reads raw CSV results and generates:
1. Summary tables with mean ± std
2. Paired t-tests between methods
3. Win rate analysis
4. Publication-ready figures

Usage:
    python3 scripts/analyze_results.py --input tabs/results.csv
    python3 scripts/analyze_results.py --input tabs/results.csv --outdir tabs --figdir figs

Requires Python 3.12+
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


METHOD_COLORS = {
    "baseline": "#1f77b4",
    "confidence_drop": "#ff7f0e",
    "fixed_anchor": "#2ca02c",
    "selective_distill": "#d62728",
    "projected_gd": "#9467bd",
    "bcwi": "#8c564b",
}

METHOD_MARKERS = {
    "baseline": "o",
    "confidence_drop": "s",
    "fixed_anchor": "^",
    "selective_distill": "D",
    "projected_gd": "P",
    "bcwi": "v",
}


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each method."""
    rows = []

    for dataset in df["dataset"].unique():
        ddf = df[df["dataset"] == dataset]

        for method in ddf["method"].unique():
            mdf = ddf[ddf["method"] == method]

            if method == "baseline":
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "param": np.nan,
                    "nfr_mean": mdf["achieved_nfr"].mean(),
                    "nfr_std": mdf["achieved_nfr"].std(),
                    "acc_mean": mdf["accuracy"].mean(),
                    "acc_std": mdf["accuracy"].std(),
                    "n_obs": len(mdf),
                })
            elif method in ["projected_gd", "bcwi"]:
                for target in mdf["target_nfr"].dropna().unique():
                    sub = mdf[mdf["target_nfr"] == target]
                    rows.append({
                        "dataset": dataset,
                        "method": method,
                        "param": target,
                        "nfr_mean": sub["achieved_nfr"].mean(),
                        "nfr_std": sub["achieved_nfr"].std(),
                        "acc_mean": sub["accuracy"].mean(),
                        "acc_std": sub["accuracy"].std(),
                        "n_obs": len(sub),
                    })
            else:
                for lam in mdf["lambda"].dropna().unique():
                    sub = mdf[mdf["lambda"] == lam]
                    rows.append({
                        "dataset": dataset,
                        "method": method,
                        "param": lam,
                        "nfr_mean": sub["achieved_nfr"].mean(),
                        "nfr_std": sub["achieved_nfr"].std(),
                        "acc_mean": sub["accuracy"].mean(),
                        "acc_std": sub["accuracy"].std(),
                        "n_obs": len(sub),
                    })

    return pd.DataFrame(rows)


def compute_method_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Rank methods by best NFR achieved while maintaining accuracy."""
    summary = compute_summary(df)
    rows = []

    for dataset in df["dataset"].unique():
        ddf = summary[summary["dataset"] == dataset]
        baseline = ddf[ddf["method"] == "baseline"]
        if baseline.empty:
            continue
        baseline_acc = baseline["acc_mean"].iloc[0]
        acc_threshold = baseline_acc - 0.01

        method_best_nfr = {}
        for method in ddf["method"].unique():
            mdf = ddf[ddf["method"] == method]
            valid = mdf[mdf["acc_mean"] >= acc_threshold]

            if len(valid) > 0:
                method_best_nfr[method] = valid["nfr_mean"].min()
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


def paired_ttests(df: pd.DataFrame) -> pd.DataFrame:
    """Perform paired t-tests between all method pairs."""
    rows = []
    methods = [m for m in df["method"].unique() if m != "baseline"]

    for dataset in df["dataset"].unique():
        ddf = df[df["dataset"] == dataset]

        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1:]:
                m1_df = ddf[ddf["method"] == method1]
                m2_df = ddf[ddf["method"] == method2]

                if m1_df.empty or m2_df.empty:
                    continue

                if "target_nfr" in m1_df.columns and m1_df["target_nfr"].notna().any():
                    params1 = m1_df["target_nfr"].dropna().unique()
                else:
                    params1 = m1_df["lambda"].dropna().unique() if "lambda" in m1_df.columns else [np.nan]

                if "target_nfr" in m2_df.columns and m2_df["target_nfr"].notna().any():
                    params2 = m2_df["target_nfr"].dropna().unique()
                else:
                    params2 = m2_df["lambda"].dropna().unique() if "lambda" in m2_df.columns else [np.nan]

                for p1 in params1:
                    for p2 in params2:
                        if np.isnan(p1):
                            sub1 = m1_df
                        elif method1 in ["projected_gd", "bcwi"]:
                            sub1 = m1_df[m1_df["target_nfr"] == p1]
                        else:
                            sub1 = m1_df[m1_df["lambda"] == p1]

                        if np.isnan(p2):
                            sub2 = m2_df
                        elif method2 in ["projected_gd", "bcwi"]:
                            sub2 = m2_df[m2_df["target_nfr"] == p2]
                        else:
                            sub2 = m2_df[m2_df["lambda"] == p2]

                        acc1 = sub1.sort_values("split")["accuracy"]
                        acc2 = sub2.sort_values("split")["accuracy"]

                        if len(acc1) == len(acc2) and len(acc1) > 1:
                            t_stat, p_value = stats.ttest_rel(acc1.values, acc2.values)
                            mean_diff = float((acc1.values - acc2.values).mean())
                            winner = "ns"
                            if p_value < 0.05:
                                winner = method1 if mean_diff > 0 else method2
                            rows.append({
                                "dataset": dataset,
                                "method1": method1,
                                "param1": p1,
                                "method2": method2,
                                "param2": p2,
                                "acc_diff": mean_diff,
                                "t_stat": float(t_stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                                "winner": winner,
                            })

    return pd.DataFrame(rows)


def plot_accuracy_delta(df: pd.DataFrame, outdir: Path) -> None:
    """Plot accuracy improvement over baseline for each method."""
    summary = compute_summary(df)

    fig, axes = plt.subplots(1, len(df["dataset"].unique()), figsize=(4 * len(df["dataset"].unique()), 5))
    axes_flat = np.atleast_1d(axes)

    for idx, dataset in enumerate(df["dataset"].unique()):
        ax = axes_flat[idx]
        ddf = summary[summary["dataset"] == dataset]

        baseline = ddf[ddf["method"] == "baseline"]
        if baseline.empty:
            continue
        baseline_acc = baseline["acc_mean"].iloc[0]

        methods_to_plot = [m for m in ddf["method"].unique() if m != "baseline"]
        x_pos = 0

        for method in methods_to_plot:
            mdf = ddf[ddf["method"] == method]
            deltas = mdf["acc_mean"].values - baseline_acc
            params = mdf["param"].values

            for i, (delta, param) in enumerate(zip(deltas, params)):
                ax.bar(x_pos, delta * 100, color=METHOD_COLORS.get(method, "gray"),
                       alpha=0.8, width=0.8)
                x_pos += 1

            x_pos += 0.5

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Accuracy Delta (%)", fontsize=10)
        ax.set_title(dataset, fontsize=12)
        ax.set_xticks([])

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS.get(m, "gray"), alpha=0.8, label=m)
        for m in METHOD_COLORS if m != "baseline"
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "accuracy_delta.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_win_rate(df: pd.DataFrame, outdir: Path) -> None:
    """Plot win rate bar chart for method rankings."""
    rankings = compute_method_rankings(df)

    win_counts = rankings[rankings["rank"] == 1].groupby("method").size()
    total = rankings["dataset"].nunique()

    methods = list(win_counts.index)
    wins = [win_counts.get(m, 0) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(len(methods)), wins, color=[METHOD_COLORS.get(m, "gray") for m in methods])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Number of #1 Rankings", fontsize=12)
    ax.set_title("Method Win Rate Across Datasets", fontsize=14)
    ax.axhline(total / len(METHOD_COLORS), color="gray", linestyle="--", label="Random")

    for bar, count in zip(bars, wins):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "win_rate.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_nfr_by_method(df: pd.DataFrame, outdir: Path) -> None:
    """Plot NFR distribution by method across all datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = [m for m in df["method"].unique() if m != "baseline"]
    positions = []
    labels = []
    pos = 0

    for method in methods:
        mdf = df[df["method"] == method]
        nfr_values = mdf["achieved_nfr"].dropna().values

        bp = ax.boxplot([nfr_values], positions=[pos], widths=0.6,
                        patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor(METHOD_COLORS.get(method, "gray"))
        bp["boxes"][0].set_alpha(0.7)

        positions.append(pos)
        labels.append(method)
        pos += 1

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Negative Flip Rate (NFR)", fontsize=12)
    ax.set_title("NFR Distribution by Method", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "nfr_by_method.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print formatted summary table to console."""
    summary = compute_summary(df)
    rankings = compute_method_rankings(df)

    print("\n" + "=" * 90)
    print("SUMMARY: Method Performance Across Datasets")
    print("=" * 90)

    print(f"\n{'Dataset':<12} {'Method':<18} {'Param':>8} {'NFR':>12} {'Accuracy':>12}")
    print("-" * 60)

    for dataset in summary["dataset"].unique():
        ddf = summary[summary["dataset"] == dataset]
        for _, row in ddf.iterrows():
            param_str = f"{row['param']:.3f}" if pd.notna(row["param"]) else "N/A"
            nfr_str = f"{row['nfr_mean']:.4f}±{row['nfr_std']:.3f}"
            acc_str = f"{row['acc_mean']:.4f}±{row['acc_std']:.3f}"
            print(f"{row['dataset']:<12} {row['method']:<18} {param_str:>8} {nfr_str:>12} {acc_str:>12}")
        print()

    print("=" * 90)
    print("METHOD RANKINGS (lower = better)")
    print("=" * 90)

    avg_rank = rankings.groupby("method")["rank"].mean().sort_values()
    win_count = rankings[rankings["rank"] == 1].groupby("method").size()

    print(f"\n{'Method':<20} {'Avg Rank':>10} {'#1 Ranks':>10}")
    print("-" * 42)
    for method in avg_rank.index:
        wins = win_count.get(method, 0)
        print(f"{method:<20} {avg_rank[method]:>10.2f} {wins:>10}")

    print("\n" + "-" * 90)
    print("Per-dataset rankings:")
    print("-" * 90)

    pivot = rankings.pivot(index="method", columns="dataset", values="rank")
    print(pivot.to_string())
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Analyze NFR experiment results")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with raw results")
    parser.add_argument("--outdir", type=Path, default=Path("tabs"), help="Output dir for tables")
    parser.add_argument("--figdir", type=Path, default=Path("figs"), help="Output dir for figures")
    args = parser.parse_args()

    print(f"Loading results from {args.input}")
    df = pd.read_csv(args.input)

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    summary = compute_summary(df)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    print(f"Saved: {args.outdir / 'summary.csv'}")

    rankings = compute_method_rankings(df)
    rankings.to_csv(args.outdir / "rankings.csv", index=False)
    print(f"Saved: {args.outdir / 'rankings.csv'}")

    ttests = paired_ttests(df)
    if not ttests.empty:
        ttests.to_csv(args.outdir / "ttests.csv", index=False)
        print(f"Saved: {args.outdir / 'ttests.csv'}")

    plot_accuracy_delta(df, args.figdir)
    print(f"Saved: {args.figdir / 'accuracy_delta.pdf'}")

    plot_win_rate(df, args.figdir)
    print(f"Saved: {args.figdir / 'win_rate.pdf'}")

    plot_nfr_by_method(df, args.figdir)
    print(f"Saved: {args.figdir / 'nfr_by_method.pdf'}")

    print_summary_table(df)


if __name__ == "__main__":
    main()
