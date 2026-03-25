#!/usr/bin/env python3
"""
Analysis script for NFR experiment results.

Reads raw CSV results and generates:
1. Summary tables with mean +/- std
2. Pareto frontier analysis
3. Publication-ready LaTeX tables (booktabs)
4. Publication-ready PDF figures (vector)

Usage:
    python3 scripts/analyze.py tabs/results.csv
    python3 scripts/analyze.py tabs/results.csv --tables-dir tabs --figures-dir figs
    python3 scripts/analyze.py tabs/results.csv --tables-only
    python3 scripts/analyze.py tabs/results.csv --figures-only

Requires Python 3.12+
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

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

METHOD_LABELS = {
    "baseline": "Baseline (ERM)",
    "confidence_drop": "Confidence Drop",
    "fixed_anchor": "Fixed Anchor",
    "selective_distill": "Selective Distill",
    "projected_gd": "Projected GD",
    "bcwi": "BCWI",
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
                    })

    return pd.DataFrame(rows)


def compute_method_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Rank methods by best NFR achieved while maintaining accuracy."""
    summary = compute_summary(df)
    rows = []

    for dataset in df["dataset"].unique():
        ddf = summary[summary["dataset"] == dataset]
        baseline_row = ddf[ddf["method"] == "baseline"]
        if baseline_row.empty:
            continue
        baseline_acc = baseline_row["acc_mean"].iloc[0]
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


def is_dominated(point: tuple[float, float], other: tuple[float, float]) -> bool:
    """Check if point is dominated by other (lower NFR, higher accuracy is better)."""
    return other[0] <= point[0] and other[1] >= point[1] and (other[0] < point[0] or other[1] > point[1])


def compute_pareto_frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """
    Return non-dominated points from a list of (nfr, accuracy, method) tuples.
    A point is on the frontier if no other point has both lower NFR and higher accuracy.
    """
    frontier = []
    for i, (nfr_i, acc_i, method_i) in enumerate(points):
        dominated = False
        for j, (nfr_j, acc_j, _) in enumerate(points):
            if i != j and is_dominated((nfr_i, acc_i), (nfr_j, acc_j)):
                dominated = True
                break
        if not dominated:
            frontier.append((nfr_i, acc_i, method_i))
    return frontier


def compute_pareto_metrics(df: pd.DataFrame) -> dict:
    """
    Compute Pareto-based summary metrics across all datasets.

    Returns dict with:
    - frontier_membership: count of frontier appearances per method
    - free_wins: count of strict dominance over baseline per method
    - avg_improvement: average NFR and accuracy change vs baseline per method
    """
    summary = compute_summary(df)
    datasets = df["dataset"].unique()
    n_datasets = len(datasets)

    frontier_counts: dict[str, int] = {}
    free_wins: dict[str, int] = {}
    nfr_improvements: dict[str, list[float]] = {}
    acc_improvements: dict[str, list[float]] = {}

    for dataset in datasets:
        ddf = summary[summary["dataset"] == dataset]
        baseline_row = ddf[ddf["method"] == "baseline"]
        if baseline_row.empty:
            continue
        baseline_nfr = baseline_row["nfr_mean"].iloc[0]
        baseline_acc = baseline_row["acc_mean"].iloc[0]

        points: list[tuple[float, float, str]] = []
        for _, row in ddf.iterrows():
            points.append((row["nfr_mean"], row["acc_mean"], row["method"]))

        frontier = compute_pareto_frontier(points)
        frontier_methods = {p[2] for p in frontier}

        for method in ddf["method"].unique():
            if method not in frontier_counts:
                frontier_counts[method] = 0
                free_wins[method] = 0
                nfr_improvements[method] = []
                acc_improvements[method] = []

            if method in frontier_methods:
                frontier_counts[method] += 1

            mdf = ddf[ddf["method"] == method]
            best_nfr = mdf["nfr_mean"].min()
            best_acc_at_best_nfr = mdf.loc[mdf["nfr_mean"].idxmin(), "acc_mean"]

            if best_nfr < baseline_nfr and best_acc_at_best_nfr > baseline_acc:
                free_wins[method] += 1

            nfr_change = (baseline_nfr - best_nfr) / baseline_nfr if baseline_nfr > 0 else 0
            acc_change = best_acc_at_best_nfr - baseline_acc
            nfr_improvements[method].append(nfr_change)
            acc_improvements[method].append(acc_change)

    return {
        "n_datasets": n_datasets,
        "frontier_membership": frontier_counts,
        "free_wins": free_wins,
        "avg_nfr_improvement": {m: np.mean(v) for m, v in nfr_improvements.items()},
        "avg_acc_improvement": {m: np.mean(v) for m, v in acc_improvements.items()},
    }


def compute_hypervolume(df: pd.DataFrame, ref_nfr: float = 1.0, ref_acc: float = 0.0) -> dict[str, float]:
    """
    Compute hypervolume indicator for each method across all datasets.

    For each dataset, compute the area dominated by the method's configurations
    relative to the reference point (ref_nfr, ref_acc).
    Returns average hypervolume per method.
    """
    summary = compute_summary(df)
    datasets = df["dataset"].unique()

    method_hvs: dict[str, list[float]] = {}

    for dataset in datasets:
        ddf = summary[summary["dataset"] == dataset]

        for method in ddf["method"].unique():
            if method not in method_hvs:
                method_hvs[method] = []

            mdf = ddf[ddf["method"] == method]
            points = [(row["nfr_mean"], row["acc_mean"]) for _, row in mdf.iterrows()]
            points = [(nfr, acc) for nfr, acc in points if nfr < ref_nfr and acc > ref_acc]

            if not points:
                method_hvs[method].append(0.0)
                continue

            points_sorted = sorted(points, key=lambda x: x[0])

            hv = 0.0
            max_acc_seen = ref_acc

            for nfr, acc in points_sorted:
                if acc > max_acc_seen:
                    width = ref_nfr - nfr
                    height = acc - max_acc_seen
                    hv += width * height
                    max_acc_seen = acc

            method_hvs[method].append(hv)

    return {m: np.mean(v) for m, v in method_hvs.items()}


def plot_pareto(df: pd.DataFrame, dataset_name: str, outdir: Path) -> None:
    """Plot Pareto frontier for a single dataset."""
    ddf = df[df["dataset"] == dataset_name]

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in ddf["method"].unique():
        mdf = ddf[ddf["method"] == method]

        if method == "baseline":
            nfr_mean = mdf["achieved_nfr"].mean()
            acc_mean = mdf["accuracy"].mean()
            ax.scatter(
                nfr_mean,
                1 - acc_mean,
                marker=METHOD_MARKERS.get(method, "o"),
                c=METHOD_COLORS.get(method, "gray"),
                s=150,
                label=METHOD_LABELS.get(method, method),
                zorder=5,
            )
        elif method in ["projected_gd", "bcwi"]:
            summary = mdf.groupby("target_nfr").agg(
                nfr_mean=("achieved_nfr", "mean"),
                nfr_std=("achieved_nfr", "std"),
                acc_mean=("accuracy", "mean"),
                acc_std=("accuracy", "std"),
            ).reset_index()
            summary = summary.sort_values("target_nfr")
            x = summary["nfr_mean"].to_numpy()
            y = 1 - summary["acc_mean"].to_numpy()
            xerr = np.nan_to_num(summary["nfr_std"].to_numpy())
            yerr = np.nan_to_num(summary["acc_std"].to_numpy())

            ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                        fmt=METHOD_MARKERS.get(method, "o"),
                        color=METHOD_COLORS.get(method, "gray"),
                        markersize=10, label=METHOD_LABELS.get(method, method), capsize=3, alpha=0.8)
            if len(x) > 1:
                ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=2)
        else:
            summary = mdf.groupby("lambda").agg(
                nfr_mean=("achieved_nfr", "mean"),
                nfr_std=("achieved_nfr", "std"),
                acc_mean=("accuracy", "mean"),
                acc_std=("accuracy", "std"),
            ).reset_index()
            summary = summary.sort_values("lambda")
            x = summary["nfr_mean"].to_numpy()
            y = 1 - summary["acc_mean"].to_numpy()
            xerr = np.nan_to_num(summary["nfr_std"].to_numpy())
            yerr = np.nan_to_num(summary["acc_std"].to_numpy())

            ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                        fmt=METHOD_MARKERS.get(method, "o"),
                        color=METHOD_COLORS.get(method, "gray"),
                        markersize=8, label=METHOD_LABELS.get(method, method), capsize=3, alpha=0.8)
            if len(x) > 1:
                ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Negative Flip Rate (NFR)")
    ax.set_ylabel("Error Rate (1 - Accuracy)")
    ax.set_title(f"{dataset_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{dataset_name}_pareto.pdf", format="pdf")
    plt.close(fig)


def plot_representative(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 2x2 grid of representative datasets for manuscript Figure 1."""
    datasets = ["adult", "heart-statlog", "diabetes", "wdbc"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for idx, name in enumerate(datasets):
        ax = axes_flat[idx]
        ddf = df[df["dataset"] == name]

        for method in ddf["method"].unique():
            mdf = ddf[ddf["method"] == method]

            if method == "baseline":
                nfr_mean = mdf["achieved_nfr"].mean()
                acc_mean = mdf["accuracy"].mean()
                ax.scatter(nfr_mean, 1 - acc_mean,
                           marker=METHOD_MARKERS.get(method, "o"),
                           c=METHOD_COLORS.get(method, "gray"),
                           s=100, zorder=5)
            elif method in ["projected_gd", "bcwi"]:
                summary = mdf.groupby("target_nfr").agg(
                    nfr_mean=("achieved_nfr", "mean"),
                    acc_mean=("accuracy", "mean"),
                ).reset_index()
                summary = summary.sort_values("target_nfr")
                x, y = summary["nfr_mean"].to_numpy(), 1 - summary["acc_mean"].to_numpy()
                ax.scatter(x, y, marker=METHOD_MARKERS.get(method, "o"),
                           c=METHOD_COLORS.get(method, "gray"), s=60, alpha=0.8)
                if len(x) > 1:
                    ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=2)
            else:
                summary = mdf.groupby("lambda").agg(
                    nfr_mean=("achieved_nfr", "mean"),
                    acc_mean=("accuracy", "mean"),
                ).reset_index()
                summary = summary.sort_values("lambda")
                x, y = summary["nfr_mean"].to_numpy(), 1 - summary["acc_mean"].to_numpy()
                ax.scatter(x, y, marker=METHOD_MARKERS.get(method, "o"),
                           c=METHOD_COLORS.get(method, "gray"), s=50, alpha=0.8)
                if len(x) > 1:
                    ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=1.5)

        ax.set_xlabel("NFR")
        ax.set_ylabel("Error Rate")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    legend_elements = [
        Line2D([0], [0], marker=METHOD_MARKERS[m], color="w",
               markerfacecolor=METHOD_COLORS[m], markersize=10,
               label=METHOD_LABELS.get(m, m))
        for m in METHOD_COLORS
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "figure1_representative.pdf", format="pdf")
    plt.close(fig)


def plot_all_datasets(df: pd.DataFrame, outdir: Path) -> None:
    """Plot combined faceted Pareto frontiers."""
    datasets = df["dataset"].unique()
    n = len(datasets)

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.atleast_2d(axes).flatten()

    for idx, name in enumerate(datasets):
        ax = axes_flat[idx]
        ddf = df[df["dataset"] == name]

        for method in ddf["method"].unique():
            mdf = ddf[ddf["method"] == method]

            if method == "baseline":
                nfr_mean = mdf["achieved_nfr"].mean()
                acc_mean = mdf["accuracy"].mean()
                ax.scatter(nfr_mean, 1 - acc_mean,
                           marker=METHOD_MARKERS.get(method, "o"),
                           c=METHOD_COLORS.get(method, "gray"),
                           s=80, zorder=5)
            elif method in ["projected_gd", "bcwi"]:
                summary = mdf.groupby("target_nfr").agg(
                    nfr_mean=("achieved_nfr", "mean"),
                    acc_mean=("accuracy", "mean"),
                ).reset_index()
                summary = summary.sort_values("target_nfr")
                x, y = summary["nfr_mean"].to_numpy(), 1 - summary["acc_mean"].to_numpy()
                ax.scatter(x, y, marker=METHOD_MARKERS.get(method, "o"),
                           c=METHOD_COLORS.get(method, "gray"), s=50, alpha=0.8)
                if len(x) > 1:
                    ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=1.5)
            else:
                summary = mdf.groupby("lambda").agg(
                    nfr_mean=("achieved_nfr", "mean"),
                    acc_mean=("accuracy", "mean"),
                ).reset_index()
                summary = summary.sort_values("lambda")
                x, y = summary["nfr_mean"].to_numpy(), 1 - summary["acc_mean"].to_numpy()
                ax.scatter(x, y, marker=METHOD_MARKERS.get(method, "o"),
                           c=METHOD_COLORS.get(method, "gray"), s=40, alpha=0.8)
                if len(x) > 1:
                    ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=1)

        ax.set_xlabel("NFR")
        ax.set_ylabel("Error")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker=METHOD_MARKERS[m], color="w",
               markerfacecolor=METHOD_COLORS[m], markersize=10,
               label=METHOD_LABELS.get(m, m))
        for m in METHOD_COLORS
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "all_datasets_pareto.pdf", format="pdf")
    plt.close(fig)


def generate_pareto_table_latex(df: pd.DataFrame, outdir: Path) -> None:
    """Generate LaTeX table for Pareto frontier membership."""
    metrics = compute_pareto_metrics(df)
    n_datasets = metrics["n_datasets"]
    frontier = metrics["frontier_membership"]

    sorted_methods = sorted(frontier.items(), key=lambda x: -x[1])

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{Pareto Frontier Membership ({n_datasets} datasets)}}")
    latex.append(r"\label{tab:pareto-frontier}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"Method & Appearances & Rate \\")
    latex.append(r"\midrule")

    for method, count in sorted_methods:
        rate = count / n_datasets
        label = METHOD_LABELS.get(method, method)
        latex.append(rf"{label} & {count}/{n_datasets} & {rate:.0%} \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pareto_frontier.tex").write_text("\n".join(latex))


def generate_rankings_table_latex(df: pd.DataFrame, outdir: Path) -> None:
    """Generate LaTeX table for method rankings."""
    rankings = compute_method_rankings(df)
    avg_rank = rankings.groupby("method")["rank"].mean().sort_values()
    win_count = rankings[rankings["rank"] == 1].groupby("method").size()
    n_datasets = rankings["dataset"].nunique()

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{Method Rankings ({n_datasets} datasets)}}")
    latex.append(r"\label{tab:rankings}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"Method & Avg Rank & \#1 Wins \\")
    latex.append(r"\midrule")

    for method in avg_rank.index:
        wins = win_count.get(method, 0)
        label = METHOD_LABELS.get(method, method)
        latex.append(rf"{label} & {avg_rank[method]:.2f} & {wins} \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "rankings.tex").write_text("\n".join(latex))


def generate_hypervolume_table_latex(df: pd.DataFrame, outdir: Path) -> None:
    """Generate LaTeX table for hypervolume metrics."""
    hypervolumes = compute_hypervolume(df)
    n_datasets = len(df["dataset"].unique())
    sorted_hv = sorted(hypervolumes.items(), key=lambda x: -x[1])

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{Hypervolume Indicator ({n_datasets} datasets)}}")
    latex.append(r"\label{tab:hypervolume}")
    latex.append(r"\begin{tabular}{lc}")
    latex.append(r"\toprule")
    latex.append(r"Method & Avg Hypervolume \\")
    latex.append(r"\midrule")

    for method, hv in sorted_hv:
        label = METHOD_LABELS.get(method, method)
        latex.append(rf"{label} & {hv:.4f} \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "hypervolume.tex").write_text("\n".join(latex))


def compute_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute head-to-head win rates between methods.

    For each dataset, compare best NFR achieved by each method
    (within 1% accuracy of baseline). Method A "wins" against B
    if A achieves lower NFR.
    """
    rankings = compute_method_rankings(df)
    datasets = rankings["dataset"].unique()
    methods = sorted(rankings["method"].unique())

    wins = {m: {n: 0 for n in methods} for m in methods}
    ties = {m: {n: 0 for n in methods} for m in methods}

    for dataset in datasets:
        ddf = rankings[rankings["dataset"] == dataset]
        method_nfr = {}
        for _, row in ddf.iterrows():
            method_nfr[row["method"]] = row["best_nfr"]

        for m1 in methods:
            for m2 in methods:
                if m1 == m2:
                    continue
                nfr1 = method_nfr.get(m1, float("inf"))
                nfr2 = method_nfr.get(m2, float("inf"))
                if np.isnan(nfr1):
                    nfr1 = float("inf")
                if np.isnan(nfr2):
                    nfr2 = float("inf")

                if nfr1 < nfr2:
                    wins[m1][m2] += 1
                elif nfr1 == nfr2:
                    ties[m1][m2] += 1

    n_datasets = len(datasets)
    rows = []
    for m1 in methods:
        row = {"method": m1}
        for m2 in methods:
            if m1 == m2:
                row[m2] = "-"
            else:
                w = wins[m1][m2]
                t = ties[m1][m2]
                l = n_datasets - w - t
                row[m2] = f"{w}-{t}-{l}"
        rows.append(row)

    return pd.DataFrame(rows)


def generate_head_to_head_latex(df: pd.DataFrame, outdir: Path) -> None:
    """Generate LaTeX table for head-to-head win rates."""
    rankings = compute_method_rankings(df)
    datasets = rankings["dataset"].unique()
    methods = ["projected_gd", "fixed_anchor", "selective_distill",
               "confidence_drop", "bcwi", "baseline"]
    n_datasets = len(datasets)

    wins = {m: {n: 0 for n in methods} for m in methods}

    for dataset in datasets:
        ddf = rankings[rankings["dataset"] == dataset]
        method_nfr = {}
        for _, row in ddf.iterrows():
            method_nfr[row["method"]] = row["best_nfr"]

        for m1 in methods:
            for m2 in methods:
                if m1 == m2:
                    continue
                nfr1 = method_nfr.get(m1, float("inf"))
                nfr2 = method_nfr.get(m2, float("inf"))
                if pd.isna(nfr1):
                    nfr1 = float("inf")
                if pd.isna(nfr2):
                    nfr2 = float("inf")

                if nfr1 < nfr2:
                    wins[m1][m2] += 1

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{Head-to-Head Wins ({n_datasets} datasets): row method vs column method}}")
    latex.append(r"\label{tab:head-to-head}")

    col_spec = "l" + "c" * len(methods)
    latex.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    header = " & ".join([METHOD_LABELS.get(m, m).split()[0] for m in methods])
    latex.append(rf" & {header} \\")
    latex.append(r"\midrule")

    for m1 in methods:
        row_label = METHOD_LABELS.get(m1, m1)
        cells = [row_label]
        for m2 in methods:
            if m1 == m2:
                cells.append("--")
            else:
                w = wins[m1][m2]
                pct = w / n_datasets * 100
                cells.append(f"{w} ({pct:.0f}\\%)")
        latex.append(" & ".join(cells) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "head_to_head.tex").write_text("\n".join(latex))


def generate_summary_table_latex(df: pd.DataFrame, outdir: Path) -> None:
    """Generate comprehensive LaTeX summary table."""
    metrics = compute_pareto_metrics(df)
    hypervolumes = compute_hypervolume(df)
    rankings = compute_method_rankings(df)

    n_datasets = metrics["n_datasets"]
    frontier = metrics["frontier_membership"]
    free_wins = metrics["free_wins"]
    avg_nfr = metrics["avg_nfr_improvement"]
    avg_acc = metrics["avg_acc_improvement"]

    avg_rank = rankings.groupby("method")["rank"].mean()
    win_count = rankings[rankings["rank"] == 1].groupby("method").size()

    methods_order = sorted(frontier.keys(), key=lambda m: -frontier[m])

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{Benchmark Summary ({n_datasets} datasets)}}")
    latex.append(r"\label{tab:summary}")
    latex.append(r"\begin{tabular}{lccccccc}")
    latex.append(r"\toprule")
    latex.append(r"Method & Frontier & Free Wins & $\Delta$NFR & $\Delta$Acc & Avg Rank & \#1 Wins & Hypervolume \\")
    latex.append(r"\midrule")

    for method in methods_order:
        label = METHOD_LABELS.get(method, method)
        fr = frontier.get(method, 0)
        fw = free_wins.get(method, 0)
        nfr_imp = avg_nfr.get(method, 0)
        acc_imp = avg_acc.get(method, 0)
        ar = avg_rank.get(method, float("nan"))
        wins = win_count.get(method, 0)
        hv = hypervolumes.get(method, 0)

        nfr_str = f"{nfr_imp:+.0%}" if not np.isnan(nfr_imp) else "---"
        acc_str = f"{acc_imp*100:+.1f}\\%" if not np.isnan(acc_imp) else "---"
        ar_str = f"{ar:.2f}" if not np.isnan(ar) else "---"

        latex.append(rf"{label} & {fr}/{n_datasets} & {fw} & {nfr_str} & {acc_str} & {ar_str} & {wins} & {hv:.4f} \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.tex").write_text("\n".join(latex))


def save_pareto_metrics_csv(df: pd.DataFrame, outdir: Path) -> None:
    """Save Pareto-based metrics to CSV."""
    metrics = compute_pareto_metrics(df)
    hypervolumes = compute_hypervolume(df)
    n_datasets = metrics["n_datasets"]

    rows = []
    for method in metrics["frontier_membership"].keys():
        rows.append({
            "method": method,
            "frontier_appearances": metrics["frontier_membership"][method],
            "frontier_rate": metrics["frontier_membership"][method] / n_datasets,
            "free_wins": metrics["free_wins"][method],
            "free_wins_rate": metrics["free_wins"][method] / n_datasets,
            "avg_nfr_improvement": metrics["avg_nfr_improvement"][method],
            "avg_acc_improvement": metrics["avg_acc_improvement"][method],
            "avg_hypervolume": hypervolumes.get(method, 0.0),
        })

    pareto_df = pd.DataFrame(rows)
    pareto_df = pareto_df.sort_values("frontier_rate", ascending=False)
    outdir.mkdir(parents=True, exist_ok=True)
    pareto_df.to_csv(outdir / "pareto_metrics.csv", index=False)


def print_summary(df: pd.DataFrame) -> None:
    """Print summary to console."""
    metrics = compute_pareto_metrics(df)
    n_datasets = metrics["n_datasets"]
    frontier = metrics["frontier_membership"]
    free_wins = metrics["free_wins"]
    avg_nfr = metrics["avg_nfr_improvement"]
    avg_acc = metrics["avg_acc_improvement"]

    hypervolumes = compute_hypervolume(df)

    print("\n" + "=" * 70)
    print(f"PARETO-BASED BENCHMARK SUMMARY ({n_datasets} datasets)")
    print("=" * 70)

    print("\n1. PARETO FRONTIER MEMBERSHIP (how often on the frontier)")
    print(f"   {'Method':<20} {'Appearances':>12} {'Rate':>8}")
    print("   " + "-" * 42)
    sorted_frontier = sorted(frontier.items(), key=lambda x: -x[1])
    for method, count in sorted_frontier:
        rate = count / n_datasets
        print(f"   {method:<20} {count:>4}/{n_datasets:<7} {rate:>7.0%}")

    print("\n2. FREE WINS (beats baseline on BOTH NFR and accuracy)")
    print(f"   {'Method':<20} {'Free Wins':>12} {'Rate':>8}")
    print("   " + "-" * 42)
    sorted_free = sorted(free_wins.items(), key=lambda x: -x[1])
    for method, count in sorted_free:
        rate = count / n_datasets
        print(f"   {method:<20} {count:>4}/{n_datasets:<7} {rate:>7.0%}")

    print("\n3. AVERAGE IMPROVEMENT vs BASELINE (at best NFR config)")
    print(f"   {'Method':<20} {'Acc Change':>12} {'NFR Change':>12}")
    print("   " + "-" * 46)
    sorted_methods = sorted(avg_nfr.keys(), key=lambda m: -avg_nfr[m])
    for method in sorted_methods:
        nfr_imp = avg_nfr[method]
        acc_imp = avg_acc[method]
        nfr_str = f"{nfr_imp:+.0%}" if not np.isnan(nfr_imp) else "N/A"
        acc_str = f"{acc_imp*100:+.1f}%" if not np.isnan(acc_imp) else "N/A"
        print(f"   {method:<20} {acc_str:>12} {nfr_str:>12}")

    print("\n4. HYPERVOLUME INDICATOR (area dominated, higher is better)")
    print(f"   {'Method':<20} {'Avg HV':>12}")
    print("   " + "-" * 34)
    sorted_hv = sorted(hypervolumes.items(), key=lambda x: -x[1])
    for method, hv in sorted_hv:
        print(f"   {method:<20} {hv:>12.4f}")

    rankings = compute_method_rankings(df)
    avg_rank = rankings.groupby("method")["rank"].mean().sort_values()
    win_count = rankings[rankings["rank"] == 1].groupby("method").size()

    print("\n5. RANKINGS (lowest NFR within 1% accuracy of baseline)")
    print(f"   {'Method':<20} {'Avg Rank':>10} {'#1 Wins':>10}")
    print("   " + "-" * 42)
    for method in avg_rank.index:
        wins = win_count.get(method, 0)
        print(f"   {method:<20} {avg_rank[method]:>10.2f} {wins:>10}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze NFR experiment results")
    parser.add_argument("input", type=Path, help="Input CSV with raw results")
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    parser.add_argument("--tables-dir", type=Path, default=repo_root / "tabs", help="Output dir for tables")
    parser.add_argument("--figures-dir", type=Path, default=repo_root / "figs", help="Output dir for figures")
    parser.add_argument("--tables-only", action="store_true", help="Only generate LaTeX tables")
    parser.add_argument("--figures-only", action="store_true", help="Only generate figures")
    args = parser.parse_args()

    print(f"Loading results from {args.input}")
    df = pd.read_csv(args.input)

    tables_dir = args.tables_dir
    figures_dir = args.figures_dir

    if not args.figures_only:
        print("\nGenerating LaTeX tables...")
        generate_pareto_table_latex(df, tables_dir)
        print(f"  Saved: {tables_dir / 'pareto_frontier.tex'}")

        generate_rankings_table_latex(df, tables_dir)
        print(f"  Saved: {tables_dir / 'rankings.tex'}")

        generate_hypervolume_table_latex(df, tables_dir)
        print(f"  Saved: {tables_dir / 'hypervolume.tex'}")

        generate_summary_table_latex(df, tables_dir)
        print(f"  Saved: {tables_dir / 'summary.tex'}")

        generate_head_to_head_latex(df, tables_dir)
        print(f"  Saved: {tables_dir / 'head_to_head.tex'}")

        h2h = compute_head_to_head(df)
        h2h.to_csv(tables_dir / "head_to_head.csv", index=False)
        print(f"  Saved: {tables_dir / 'head_to_head.csv'}")

        summary = compute_summary(df)
        summary.to_csv(tables_dir / "summary.csv", index=False)
        print(f"  Saved: {tables_dir / 'summary.csv'}")

        rankings = compute_method_rankings(df)
        rankings.to_csv(tables_dir / "rankings.csv", index=False)
        print(f"  Saved: {tables_dir / 'rankings.csv'}")

        save_pareto_metrics_csv(df, tables_dir)
        print(f"  Saved: {tables_dir / 'pareto_metrics.csv'}")

    if not args.tables_only:
        print("\nGenerating figures...")
        for name in df["dataset"].unique():
            plot_pareto(df, name, figures_dir)
            print(f"  Saved: {figures_dir / f'{name}_pareto.pdf'}")

        plot_all_datasets(df, figures_dir)
        print(f"  Saved: {figures_dir / 'all_datasets_pareto.pdf'}")

        plot_representative(df, figures_dir)
        print(f"  Saved: {figures_dir / 'figure1_representative.pdf'}")

    print_summary(df)


if __name__ == "__main__":
    main()
