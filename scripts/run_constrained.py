#!/usr/bin/env python3
"""
Benchmark runner: run all methods across datasets and generate results.

This compares 6 methods for preventing model regression:
1. baseline - Standard ERM (no NFR control)
2. confidence_drop - Epoch-local loss increase penalty
3. fixed_anchor - Uses incumbent loss as anchor on fixed set
4. selective_distill - Distills to incumbent predictions
5. projected_gd - Projected gradient descent with NFR constraint
6. bcwi - Post-hoc weight interpolation

Usage:
    # Quick test (2 datasets, 3 splits)
    python3 scripts/run_constrained.py --datasets adult,diabetes --n-splits 3

    # Full benchmark (all 5 datasets, 10 splits)
    python3 scripts/run_constrained.py --n-splits 10

    # Analyze existing results
    python3 scripts/run_constrained.py --analyze-only --input tabs/results.csv

Requires Python 3.12+
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy import stats

from datasets import BenchmarkDataset, list_datasets, load_dataset
from metrics import evaluate
from models import MLP, TrainConfig, set_seed, train_erm
from training import (
    train_baseline,
    train_confidence_drop,
    train_fixed_anchor,
    train_selective_distill,
    train_projected_gd,
    bcwi_select,
)


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

LAMBDA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
NFR_TARGETS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for running experiments."""
    n_splits: int = 10
    lambda_values: list[float] = field(default_factory=lambda: LAMBDA_VALUES)
    nfr_targets: list[float] = field(default_factory=lambda: NFR_TARGETS)
    epochs_baseline: int = 50
    epochs_constrained: int = 100
    lr: float = 0.001
    batch_size: int = 256
    seed_base: int = 200
    warmup_epochs: int = 10


@dataclass(slots=True)
class SplitData:
    """Data for a single train/val/test split."""
    x_old: np.ndarray
    y_old: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_anchor: np.ndarray
    y_anchor: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def create_split(dataset: BenchmarkDataset, seed: int) -> SplitData:
    """Create a random train/val/test split with anchor set."""
    set_seed(seed)

    n_train = len(dataset.x_train)
    n_test = len(dataset.x_test)

    old_n = n_train // 3
    anchor_n = n_test // 5
    val_n = n_test // 5
    test_n = n_test - anchor_n - val_n

    train_perm = np.random.permutation(n_train)
    test_perm = np.random.permutation(n_test)

    old_idx = train_perm[:old_n]

    anchor_idx = test_perm[:anchor_n]
    val_idx = test_perm[anchor_n:anchor_n + val_n]
    test_idx = test_perm[anchor_n + val_n:]

    x_old, y_old = dataset.x_train[old_idx], dataset.y_train[old_idx]
    x_train = dataset.x_train[train_perm]
    y_train = dataset.y_train[train_perm]

    return SplitData(
        x_old=x_old, y_old=y_old,
        x_train=x_train, y_train=y_train,
        x_anchor=dataset.x_test[anchor_idx], y_anchor=dataset.y_test[anchor_idx],
        x_val=dataset.x_test[val_idx], y_val=dataset.y_test[val_idx],
        x_test=dataset.x_test[test_idx], y_test=dataset.y_test[test_idx],
    )


def run_split(
    dataset: BenchmarkDataset,
    split_idx: int,
    config: ExperimentConfig,
) -> list[dict]:
    """Run all methods on a single split."""
    seed = config.seed_base + split_idx
    split = create_split(dataset, seed)
    input_dim = dataset.n_features

    baseline_config = TrainConfig(
        epochs=config.epochs_baseline,
        lr=config.lr,
        batch_size=config.batch_size,
        seed=seed,
    )

    incumbent = train_erm(split.x_old, split.y_old, input_dim, baseline_config)

    x_anchor_t = torch.tensor(split.x_anchor, dtype=torch.float32)
    x_val_t = torch.tensor(split.x_val, dtype=torch.float32)
    x_test_t = torch.tensor(split.x_test, dtype=torch.float32)

    incumbent_anchor_prob = incumbent.predict_prob(x_anchor_t).numpy()
    incumbent_val_prob = incumbent.predict_prob(x_val_t).numpy()
    incumbent_test_prob = incumbent.predict_prob(x_test_t).numpy()

    incumbent_anchor_correct = (incumbent_anchor_prob >= 0.5).astype(int) == split.y_anchor
    x_anchor_keep = split.x_anchor[incumbent_anchor_correct]
    y_anchor_keep = split.y_anchor[incumbent_anchor_correct]
    incumbent_anchor_keep_prob = incumbent_anchor_prob[incumbent_anchor_correct]

    rows: list[dict] = []

    _, baseline_eval = train_baseline(
        split.x_train, split.y_train,
        split.x_test, split.y_test,
        incumbent_test_prob,
        TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 1),
    )
    rows.append({
        "dataset": dataset.name,
        "split": split_idx,
        "method": "baseline",
        "lambda": np.nan,
        "target_nfr": np.nan,
        "achieved_nfr": baseline_eval.nfr,
        "accuracy": baseline_eval.accuracy,
        "pfr": baseline_eval.pfr,
    })

    for lam in config.lambda_values:
        _, cd_eval, _ = train_confidence_drop(
            split.x_train, split.y_train,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_val_prob, incumbent_test_prob,
            lam=lam,
            config=TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 2),
            warmup_epochs=config.warmup_epochs,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "confidence_drop",
            "lambda": lam,
            "target_nfr": np.nan,
            "achieved_nfr": cd_eval.nfr,
            "accuracy": cd_eval.accuracy,
            "pfr": cd_eval.pfr,
        })

        _, fa_eval, _ = train_fixed_anchor(
            incumbent,
            split.x_train, split.y_train,
            x_anchor_keep, y_anchor_keep,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_anchor_keep_prob,
            incumbent_val_prob, incumbent_test_prob,
            lam=lam,
            config=TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 3),
            warmup_epochs=config.warmup_epochs,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "fixed_anchor",
            "lambda": lam,
            "target_nfr": np.nan,
            "achieved_nfr": fa_eval.nfr,
            "accuracy": fa_eval.accuracy,
            "pfr": fa_eval.pfr,
        })

        _, sd_eval, _ = train_selective_distill(
            incumbent,
            split.x_train, split.y_train,
            x_anchor_keep,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_anchor_keep_prob,
            incumbent_val_prob, incumbent_test_prob,
            lam=lam,
            config=TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 4),
            warmup_epochs=config.warmup_epochs,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "selective_distill",
            "lambda": lam,
            "target_nfr": np.nan,
            "achieved_nfr": sd_eval.nfr,
            "accuracy": sd_eval.accuracy,
            "pfr": sd_eval.pfr,
        })

    for target_nfr in config.nfr_targets:
        _, pgd_eval, _ = train_projected_gd(
            incumbent,
            split.x_train, split.y_train,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_val_prob, incumbent_test_prob,
            target_nfr=target_nfr,
            config=TrainConfig(epochs=config.epochs_constrained, lr=config.lr, batch_size=config.batch_size, seed=seed + 5),
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "projected_gd",
            "lambda": np.nan,
            "target_nfr": target_nfr,
            "achieved_nfr": pgd_eval.nfr,
            "accuracy": pgd_eval.accuracy,
            "pfr": pgd_eval.pfr,
        })

        _, bcwi_eval, _ = bcwi_select(
            incumbent,
            split.x_train, split.y_train,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_val_prob, incumbent_test_prob,
            target_nfr=target_nfr,
            config=TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 6),
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "bcwi",
            "lambda": np.nan,
            "target_nfr": target_nfr,
            "achieved_nfr": bcwi_eval.nfr,
            "accuracy": bcwi_eval.accuracy,
            "pfr": bcwi_eval.pfr,
        })

    return rows


def run_benchmark(
    datasets: list[BenchmarkDataset],
    config: ExperimentConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run experiment across multiple datasets."""
    all_rows: list[dict] = []

    for dataset in datasets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset.name}")
            print(f"  Train: {dataset.x_train.shape}, Test: {dataset.x_test.shape}")
            print(f"{'='*60}")

        for split_idx in range(config.n_splits):
            if verbose:
                print(f"  Split {split_idx + 1}/{config.n_splits}")
            rows = run_split(dataset, split_idx, config)
            all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def plot_pareto(df: pd.DataFrame, dataset_name: str, outdir: Path) -> None:
    """Plot Pareto frontier for a single dataset."""
    ddf = df[df["dataset"] == dataset_name]

    fig, ax = plt.subplots(figsize=(10, 7))

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
                label=method,
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
                        markersize=10, label=method, capsize=3, alpha=0.8)
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
                        markersize=8, label=method, capsize=3, alpha=0.8)
            if len(x) > 1:
                ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Negative Flip Rate (NFR)", fontsize=12)
    ax.set_ylabel("Error Rate (1 - Accuracy)", fontsize=12)
    ax.set_title(f"{dataset_name}: All Methods Pareto Frontiers", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{dataset_name}_pareto.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_all_datasets(df: pd.DataFrame, outdir: Path) -> None:
    """Plot combined faceted Pareto frontiers."""
    datasets = df["dataset"].unique()
    n = len(datasets)

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
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
                    ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=1.5)
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
                    ax.plot(x, y, "-", color=METHOD_COLORS.get(method, "gray"), alpha=0.5, linewidth=1)

        ax.set_xlabel("NFR", fontsize=10)
        ax.set_ylabel("Error", fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker=METHOD_MARKERS[m], color="w",
               markerfacecolor=METHOD_COLORS[m], markersize=10, label=m)
        for m in METHOD_COLORS
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "all_datasets_pareto.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)


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
        baseline_acc = ddf[ddf["method"] == "baseline"]["acc_mean"].iloc[0]
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


def print_summary(df: pd.DataFrame) -> None:
    """Print summary to console."""
    summary = compute_summary(df)
    rankings = compute_method_rankings(df)

    print("\n" + "=" * 80)
    print("SUMMARY: All Methods Comparison")
    print("=" * 80)

    avg_rank = rankings.groupby("method")["rank"].mean().sort_values()
    win_count = rankings[rankings["rank"] == 1].groupby("method").size()

    print(f"\n{'Method':<20} {'Avg Rank':>10} {'#1 Ranks':>10}")
    print("-" * 42)
    for method in avg_rank.index:
        wins = win_count.get(method, 0)
        print(f"{method:<20} {avg_rank[method]:>10.2f} {wins:>10}")

    print("\n" + "-" * 80)
    print("Per-dataset rankings (lower = better):")
    print("-" * 80)

    pivot = rankings.pivot(index="method", columns="dataset", values="rank")
    print(pivot.to_string())
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: All Methods Comparison")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated datasets (default: all)")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of splits per dataset")
    parser.add_argument("--epochs-baseline", type=int, default=50, help="Epochs for baseline methods")
    parser.add_argument("--epochs-constrained", type=int, default=100, help="Epochs for constrained methods")
    parser.add_argument("--outdir", type=Path, default=Path("tabs"), help="Output dir for CSVs")
    parser.add_argument("--figdir", type=Path, default=Path("figs"), help="Output dir for figures")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results")
    parser.add_argument("--input", type=Path, default=None, help="Input CSV for analyze-only mode")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        if args.input is None:
            args.input = args.outdir / "results.csv"
        print(f"Loading results from {args.input}")
        df = pd.read_csv(args.input)
    else:
        dataset_names = list_datasets() if args.datasets is None else [d.strip() for d in args.datasets.split(",")]
        datasets = [load_dataset(name) for name in dataset_names]

        config = ExperimentConfig(
            n_splits=args.n_splits,
            epochs_baseline=args.epochs_baseline,
            epochs_constrained=args.epochs_constrained,
        )

        df = run_benchmark(datasets, config, verbose=True)
        df.to_csv(args.outdir / "results.csv", index=False)
        print(f"\nSaved: {args.outdir / 'results.csv'}")

    for name in df["dataset"].unique():
        plot_pareto(df, name, args.figdir)
        print(f"Saved: {args.figdir / f'{name}_pareto.pdf'}")

    plot_all_datasets(df, args.figdir)
    print(f"Saved: {args.figdir / 'all_datasets_pareto.pdf'}")

    summary = compute_summary(df)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    print(f"Saved: {args.outdir / 'summary.csv'}")

    rankings = compute_method_rankings(df)
    rankings.to_csv(args.outdir / "rankings.csv", index=False)
    print(f"Saved: {args.outdir / 'rankings.csv'}")

    print_summary(df)


if __name__ == "__main__":
    main()
