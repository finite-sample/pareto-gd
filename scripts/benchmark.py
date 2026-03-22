#!/usr/bin/env python3
"""
Benchmark runner: run methods across datasets and generate results.

This is the main entry point for running experiments.

Usage:
    # Quick test (2 datasets, 3 splits)
    python3 scripts/benchmark.py --datasets adult,diabetes --n-splits 3

    # Full benchmark (all 5 datasets, 10 splits)
    python3 scripts/benchmark.py --n-splits 10

    # Analyze existing results
    python3 scripts/benchmark.py --analyze-only --input tabs/results.csv

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
from method_bcwi import bcwi
from method_projected_gd import projected_gd
from models import MLP, TrainConfig, set_seed, train_erm


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for running experiments."""
    n_splits: int = 10
    nfr_targets: list[float] = field(default_factory=lambda: [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
    epochs_baseline: int = 50
    epochs_constrained: int = 100
    lr: float = 0.001
    batch_size: int = 256
    seed_base: int = 200


@dataclass(slots=True)
class SplitData:
    """Data for a single train/val/test split."""
    x_old: np.ndarray
    y_old: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def create_split(dataset: BenchmarkDataset, seed: int) -> SplitData:
    """Create a random train/val/test split."""
    set_seed(seed)

    n_train = len(dataset.x_train)
    n_test = len(dataset.x_test)

    old_n = n_train // 3
    val_n = n_test // 3

    train_perm = np.random.permutation(n_train)
    test_perm = np.random.permutation(n_test)

    old_idx = train_perm[:old_n]
    val_idx = test_perm[:val_n]
    test_idx = test_perm[val_n:]

    x_old, y_old = dataset.x_train[old_idx], dataset.y_train[old_idx]
    x_train = dataset.x_train[train_perm]
    y_train = dataset.y_train[train_perm]

    return SplitData(
        x_old=x_old, y_old=y_old,
        x_train=x_train, y_train=y_train,
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

    x_val_t = torch.tensor(split.x_val, dtype=torch.float32)
    x_test_t = torch.tensor(split.x_test, dtype=torch.float32)
    incumbent_val_prob = incumbent.predict_prob(x_val_t).numpy()
    incumbent_test_prob = incumbent.predict_prob(x_test_t).numpy()

    rows: list[dict] = []

    candidate = train_erm(split.x_train, split.y_train, input_dim, baseline_config)
    baseline_eval = evaluate(candidate, split.x_test, split.y_test, incumbent_test_prob)
    rows.append({
        "dataset": dataset.name,
        "split": split_idx,
        "method": "baseline",
        "target_nfr": np.nan,
        "achieved_nfr": baseline_eval.nfr,
        "accuracy": baseline_eval.accuracy,
        "pfr": baseline_eval.pfr,
    })

    for target_nfr in config.nfr_targets:
        bcwi_config = TrainConfig(
            epochs=config.epochs_baseline,
            lr=config.lr,
            batch_size=config.batch_size,
            seed=seed + 1,
        )
        _, bcwi_eval, _ = bcwi(
            incumbent, split.x_train, split.y_train,
            split.x_val, split.y_val, split.x_test, split.y_test,
            incumbent_val_prob, incumbent_test_prob,
            target_nfr, bcwi_config,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "bcwi",
            "target_nfr": target_nfr,
            "achieved_nfr": bcwi_eval.nfr,
            "accuracy": bcwi_eval.accuracy,
            "pfr": bcwi_eval.pfr,
        })

        pgd_config = TrainConfig(
            epochs=config.epochs_constrained,
            lr=config.lr,
            batch_size=config.batch_size,
            seed=seed + 2,
        )
        _, pgd_eval, _ = projected_gd(
            incumbent, split.x_train, split.y_train,
            split.x_val, split.y_val, split.x_test, split.y_test,
            incumbent_val_prob, incumbent_test_prob,
            target_nfr, pgd_config,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "constrained",
            "target_nfr": target_nfr,
            "achieved_nfr": pgd_eval.nfr,
            "accuracy": pgd_eval.accuracy,
            "pfr": pgd_eval.pfr,
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
    colors = {"bcwi": "#9467bd", "constrained": "#2ca02c", "baseline": "#1f77b4"}
    markers = {"bcwi": "v", "constrained": "P", "baseline": "o"}

    ddf = df[df["dataset"] == dataset_name]
    summary = ddf.groupby(["method", "target_nfr"]).agg(
        nfr_mean=("achieved_nfr", "mean"),
        nfr_std=("achieved_nfr", "std"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))

    baseline = summary[summary["method"] == "baseline"]
    if not baseline.empty:
        ax.scatter(
            baseline["nfr_mean"].iloc[0],
            1 - baseline["acc_mean"].iloc[0],
            marker=markers["baseline"],
            c=colors["baseline"],
            s=150,
            label="baseline",
            zorder=5,
        )

    for method in ["bcwi", "constrained"]:
        mdf = summary[summary["method"] == method].dropna(subset=["nfr_mean", "acc_mean"])
        if mdf.empty:
            continue

        mdf = mdf.sort_values("target_nfr")
        x = mdf["nfr_mean"].to_numpy()
        y = 1 - mdf["acc_mean"].to_numpy()
        xerr = np.nan_to_num(mdf["nfr_std"].to_numpy())
        yerr = np.nan_to_num(mdf["acc_std"].to_numpy())

        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    fmt=markers[method], color=colors[method],
                    markersize=10, label=method, capsize=3, alpha=0.8)

        if len(x) > 1:
            ax.plot(x, y, "-", color=colors[method], alpha=0.5, linewidth=2)

    ax.set_xlabel("Negative Flip Rate (NFR)", fontsize=12)
    ax.set_ylabel("Error Rate (1 - Accuracy)", fontsize=12)
    ax.set_title(f"{dataset_name}: BCWI vs Constrained", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
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

    colors = {"bcwi": "#9467bd", "constrained": "#2ca02c", "baseline": "#1f77b4"}
    markers = {"bcwi": "v", "constrained": "P", "baseline": "o"}

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.atleast_2d(axes).flatten()

    for idx, name in enumerate(datasets):
        ax = axes_flat[idx]
        ddf = df[df["dataset"] == name]
        summary = ddf.groupby(["method", "target_nfr"]).agg(
            nfr_mean=("achieved_nfr", "mean"),
            acc_mean=("accuracy", "mean"),
        ).reset_index()

        baseline = summary[summary["method"] == "baseline"]
        if not baseline.empty:
            ax.scatter(baseline["nfr_mean"].iloc[0], 1 - baseline["acc_mean"].iloc[0],
                       marker=markers["baseline"], c=colors["baseline"], s=100, zorder=5)

        for method in ["bcwi", "constrained"]:
            mdf = summary[summary["method"] == method].dropna(subset=["nfr_mean", "acc_mean"])
            if mdf.empty:
                continue
            mdf = mdf.sort_values("target_nfr")
            x, y = mdf["nfr_mean"].to_numpy(), 1 - mdf["acc_mean"].to_numpy()
            ax.scatter(x, y, marker=markers[method], c=colors[method], s=60, alpha=0.8)
            if len(x) > 1:
                ax.plot(x, y, "-", color=colors[method], alpha=0.5, linewidth=1.5)

        ax.set_xlabel("NFR", fontsize=10)
        ax.set_ylabel("Error", fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor=c, markersize=10, label=n)
        for n, m, c in [("baseline", "o", colors["baseline"]),
                        ("bcwi", "v", colors["bcwi"]),
                        ("constrained", "P", colors["constrained"])]
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "all_datasets_pareto.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)


def compute_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head comparison."""
    summary = df.groupby(["dataset", "method", "target_nfr"]).agg(
        nfr_mean=("achieved_nfr", "mean"),
        acc_mean=("accuracy", "mean"),
    ).reset_index()

    bcwi_df = summary[summary["method"] == "bcwi"][["dataset", "target_nfr", "nfr_mean", "acc_mean"]]
    bcwi_df = bcwi_df.rename(columns={"nfr_mean": "bcwi_nfr", "acc_mean": "bcwi_acc"})

    const_df = summary[summary["method"] == "constrained"][["dataset", "target_nfr", "nfr_mean", "acc_mean"]]
    const_df = const_df.rename(columns={"nfr_mean": "const_nfr", "acc_mean": "const_acc"})

    comp = bcwi_df.merge(const_df, on=["dataset", "target_nfr"], how="outer")
    comp["delta_acc"] = comp["const_acc"] - comp["bcwi_acc"]
    comp["winner"] = np.where(comp["delta_acc"] > 0, "constrained",
                              np.where(comp["delta_acc"] < 0, "bcwi", "tie"))
    return comp


def paired_ttests(df: pd.DataFrame) -> pd.DataFrame:
    """Perform paired t-tests between constrained and BCWI."""
    rows: list[dict] = []

    for dataset in df["dataset"].unique():
        ddf = df[df["dataset"] == dataset]
        targets = ddf[ddf["target_nfr"].notna()]["target_nfr"].unique()

        for target in targets:
            bcwi_acc = ddf[(ddf["method"] == "bcwi") & (ddf["target_nfr"] == target)].sort_values("split")["accuracy"]
            const_acc = ddf[(ddf["method"] == "constrained") & (ddf["target_nfr"] == target)].sort_values("split")["accuracy"]

            if len(bcwi_acc) == len(const_acc) and len(bcwi_acc) > 1:
                t_stat, p_value = stats.ttest_rel(const_acc.values, bcwi_acc.values)
                mean_diff = float((const_acc.values - bcwi_acc.values).mean())
                winner = "ns"
                if p_value < 0.05:
                    winner = "constrained" if mean_diff > 0 else "bcwi"
                rows.append({
                    "dataset": dataset,
                    "target_nfr": target,
                    "mean_diff": mean_diff,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "winner": winner,
                })

    return pd.DataFrame(rows)


def print_summary(comp: pd.DataFrame) -> None:
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("Summary: Constrained vs BCWI")
    print("=" * 70)

    print(f"\n{'Dataset':<12} {'Target':<8} {'BCWI NFR':<10} {'BCWI Acc':<10} {'Const NFR':<10} {'Const Acc':<10} {'Winner':<12}")
    print("-" * 70)

    for _, row in comp.iterrows():
        print(f"{row['dataset']:<12} {row['target_nfr']:<8.3f} {row['bcwi_nfr']:<10.4f} {row['bcwi_acc']:<10.4f} {row['const_nfr']:<10.4f} {row['const_acc']:<10.4f} {row['winner']:<12}")

    const_wins = (comp["winner"] == "constrained").sum()
    bcwi_wins = (comp["winner"] == "bcwi").sum()
    ties = (comp["winner"] == "tie").sum()
    total = len(comp)

    print("\n" + "=" * 70)
    print(f"Win Rate: Constrained {const_wins}/{total} ({100*const_wins/total:.1f}%), BCWI {bcwi_wins}/{total} ({100*bcwi_wins/total:.1f}%), Ties {ties}")
    print(f"Mean Accuracy Delta: {comp['delta_acc'].mean():+.4f} (±{comp['delta_acc'].std():.4f})")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Constrained vs BCWI")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated datasets (default: all)")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of splits per dataset")
    parser.add_argument("--epochs-baseline", type=int, default=50, help="Epochs for baseline/BCWI")
    parser.add_argument("--epochs-constrained", type=int, default=100, help="Epochs for constrained")
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

    comp = compute_comparison(df)
    comp.to_csv(args.outdir / "comparison.csv", index=False)
    print(f"Saved: {args.outdir / 'comparison.csv'}")

    ttests = paired_ttests(df)
    ttests.to_csv(args.outdir / "ttests.csv", index=False)
    print(f"Saved: {args.outdir / 'ttests.csv'}")

    print_summary(comp)


if __name__ == "__main__":
    main()
