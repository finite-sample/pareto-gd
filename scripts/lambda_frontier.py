#!/usr/bin/env python3
"""
λ-Frontier Sweep: Pareto frontiers for forgetting-penalized training.

Sweeps λ over a grid for each method and produces:
1. tabs/{dataset}_lambda_frontier_raw.csv - all (method, λ, split, metrics) rows
2. tabs/{dataset}_lambda_frontier_summary.csv - mean ± std per (method, λ)
3. figs/{dataset}_pareto_frontier.pdf - accuracy vs NFR plot with frontiers
4. figs/{dataset}_pareto_frontier_with_cumforgetting.pdf - NFR vs cumulative forgetting

Supports multiple datasets via --dataset flag or use run_benchmark.py for full sweep.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forgetting_pilot import (
    Metrics,
    bcwi_select,
    evaluate_candidate,
    fit_baseline_epochs,
    fit_confidence_drop,
    fit_fixed_anchor,
    fit_selective_distill,
    load_adult,
    predict_prob,
    set_seed,
    summarize,
    transform_data,
)

from datasets import BenchmarkDataset, load_dataset, list_datasets


LAMBDAS_TRAIN = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RHOS_BCWI = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

METHOD_COLORS = {
    "baseline": "#1f77b4",
    "confidence_drop": "#ff7f0e",
    "fixed_anchor": "#2ca02c",
    "selective_distill": "#d62728",
    "bcwi": "#9467bd",
}

METHOD_MARKERS = {
    "baseline": "o",
    "confidence_drop": "s",
    "fixed_anchor": "^",
    "selective_distill": "D",
    "bcwi": "v",
}

DATASET_SAMPLE_SIZES = {
    "adult": {"old_n": 5000, "new_n": 8000, "anchor_n": 1500, "val_n": 2500, "test_n": 6000},
    "bank": {"old_n": 5000, "new_n": 8000, "anchor_n": 1500, "val_n": 2500, "test_n": 6000},
    "credit": {"old_n": 3000, "new_n": 5000, "anchor_n": 1000, "val_n": 1500, "test_n": 3000},
    "diabetes": {"old_n": 150, "new_n": 200, "anchor_n": 50, "val_n": 50, "test_n": 80},
    "spambase": {"old_n": 800, "new_n": 1200, "anchor_n": 250, "val_n": 400, "test_n": 500},
}

DEFAULT_SIZES = {"old_n": 2000, "new_n": 3000, "anchor_n": 500, "val_n": 800, "test_n": 1500}


def get_sample_sizes(dataset_name: str, n_train: int, n_test: int) -> dict:
    """Get appropriate sample sizes for a dataset, with fallback scaling."""
    if dataset_name in DATASET_SAMPLE_SIZES:
        sizes = DATASET_SAMPLE_SIZES[dataset_name].copy()
    else:
        sizes = DEFAULT_SIZES.copy()

    total_train_needed = sizes["old_n"] + sizes["new_n"]
    total_test_needed = sizes["anchor_n"] + sizes["val_n"] + sizes["test_n"]

    if total_train_needed > n_train:
        scale = n_train / total_train_needed * 0.9
        sizes["old_n"] = int(sizes["old_n"] * scale)
        sizes["new_n"] = int(sizes["new_n"] * scale)

    if total_test_needed > n_test:
        scale = n_test / total_test_needed * 0.9
        sizes["anchor_n"] = int(sizes["anchor_n"] * scale)
        sizes["val_n"] = int(sizes["val_n"] * scale)
        sizes["test_n"] = int(sizes["test_n"] * scale)

    return sizes


def run_lambda_sweep(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    lambdas: List[float],
    rhos: List[float],
    n_splits: int = 5,
    old_n: int = 5000,
    new_n: int = 8000,
    anchor_n: int = 1500,
    val_n: int = 2500,
    test_n: int = 6000,
) -> pd.DataFrame:
    """Run all methods over λ grid."""
    rows = []

    for split in range(n_splits):
        seed = 100 + split
        set_seed(seed)
        print(f"  Split {split + 1}/{n_splits}")

        train_idx = np.arange(len(x_train))
        test_idx = np.arange(len(x_test))

        train_sub = np.random.choice(train_idx, size=old_n + new_n, replace=False)
        old_idx = train_sub[:old_n]
        new_idx = train_sub[old_n:]

        test_sub = np.random.choice(test_idx, size=anchor_n + val_n + test_n, replace=False)
        anchor_idx = test_sub[:anchor_n]
        val_idx = test_sub[anchor_n : anchor_n + val_n]
        test_eval_idx = test_sub[anchor_n + val_n :]

        x_old, y_old = x_train[old_idx], y_train[old_idx]
        x_new, y_new = x_train[new_idx], y_train[new_idx]
        x_cand = np.vstack([x_old, x_new])
        y_cand = np.concatenate([y_old, y_new])

        x_anchor, y_anchor = x_test[anchor_idx], y_test[anchor_idx]
        x_val, y_val = x_test[val_idx], y_test[val_idx]
        x_eval, y_eval = x_test[test_eval_idx], y_test[test_eval_idx]

        incumbent, _, _ = fit_baseline_epochs(
            x_old, y_old, x_anchor, y_anchor, epochs=20, seed=seed
        )
        incumbent_anchor_prob = predict_prob(incumbent, x_anchor)
        incumbent_val_prob = predict_prob(incumbent, x_val)
        incumbent_eval_prob = predict_prob(incumbent, x_eval)

        incumbent_anchor_correct = (incumbent_anchor_prob >= 0.5).astype(int) == y_anchor
        x_anchor_keep = x_anchor[incumbent_anchor_correct]
        y_anchor_keep = y_anchor[incumbent_anchor_correct]
        incumbent_anchor_keep_prob = incumbent_anchor_prob[incumbent_anchor_correct]

        cand_base, cf_base, _ = fit_baseline_epochs(
            x_cand, y_cand, x_anchor, y_anchor, epochs=20, seed=seed + 1
        )
        base_eval = evaluate_candidate(
            predict_prob(cand_base, x_eval), incumbent_eval_prob, y_eval, cf_base
        )
        rows.append(
            {
                "split": split,
                "method": "baseline",
                "lam": 0.0,
                **asdict(base_eval),
            }
        )

        for lam in lambdas:
            cand_cd, cf_cd = fit_confidence_drop(
                x_cand,
                y_cand,
                x_anchor,
                y_anchor,
                lam=lam,
                epochs=20,
                seed=seed + 2,
            )
            cd_eval = evaluate_candidate(
                predict_prob(cand_cd, x_eval), incumbent_eval_prob, y_eval, cf_cd
            )
            rows.append(
                {
                    "split": split,
                    "method": "confidence_drop",
                    "lam": lam,
                    **asdict(cd_eval),
                }
            )

        for lam in lambdas:
            cand_fa, cf_fa = fit_fixed_anchor(
                x_cand,
                y_cand,
                x_anchor_keep,
                y_anchor_keep,
                incumbent_anchor_keep_prob,
                x_anchor,
                y_anchor,
                lam=lam,
                epochs=20,
                seed=seed + 3,
            )
            fa_eval = evaluate_candidate(
                predict_prob(cand_fa, x_eval), incumbent_eval_prob, y_eval, cf_fa
            )
            rows.append(
                {
                    "split": split,
                    "method": "fixed_anchor",
                    "lam": lam,
                    **asdict(fa_eval),
                }
            )

        for lam in lambdas:
            cand_sd, cf_sd = fit_selective_distill(
                x_cand,
                y_cand,
                x_anchor_keep,
                incumbent_anchor_keep_prob,
                x_anchor,
                y_anchor,
                lam=lam,
                epochs=20,
                seed=seed + 4,
            )
            sd_eval = evaluate_candidate(
                predict_prob(cand_sd, x_eval), incumbent_eval_prob, y_eval, cf_sd
            )
            rows.append(
                {
                    "split": split,
                    "method": "selective_distill",
                    "lam": lam,
                    **asdict(sd_eval),
                }
            )

        for rho in rhos:
            bcwi_clf, alpha = bcwi_select(
                incumbent, cand_base, x_val, y_val, incumbent_val_prob, rho=rho
            )
            bcwi_eval = evaluate_candidate(
                predict_prob(bcwi_clf, x_eval),
                incumbent_eval_prob,
                y_eval,
                cum_forgetting=0,
            )
            rows.append(
                {
                    "split": split,
                    "method": "bcwi",
                    "lam": rho,
                    "alpha": alpha,
                    **asdict(bcwi_eval),
                }
            )

    return pd.DataFrame(rows)


def compute_pareto_frontier(
    points: List[Tuple[float, float]], minimize_x: bool = True, minimize_y: bool = True
) -> List[int]:
    """Return indices of Pareto-optimal points."""
    if not points:
        return []

    n = len(points)
    is_dominated = [False] * n

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            xi, yi = points[i]
            xj, yj = points[j]

            if minimize_x:
                x_better = xj <= xi
                x_strictly = xj < xi
            else:
                x_better = xj >= xi
                x_strictly = xj > xi

            if minimize_y:
                y_better = yj <= yi
                y_strictly = yj < yi
            else:
                y_better = yj >= yi
                y_strictly = yj > yi

            if x_better and y_better and (x_strictly or y_strictly):
                is_dominated[i] = True
                break

    return [i for i in range(n) if not is_dominated[i]]


def plot_pareto_frontiers(df: pd.DataFrame, outdir: Path, dataset_name: str = "") -> None:
    """Generate accuracy vs NFR plot with method frontiers."""
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

        pts = list(zip(x, y))
        pareto_idx = compute_pareto_frontier(pts, minimize_x=True, minimize_y=True)
        if len(pareto_idx) > 1:
            pareto_points = sorted([(x[i], y[i]) for i in pareto_idx])
            px, py = zip(*pareto_points)
            ax.plot(px, py, "-", color=color, alpha=0.5, linewidth=2)

    ax.set_xlabel("Negative Flip Rate (NFR)", fontsize=12)
    ax.set_ylabel("Error Rate (1 - Accuracy)", fontsize=12)
    title = "Pareto Frontiers: Stability vs Accuracy"
    if dataset_name:
        title = f"{dataset_name.upper()}: {title}"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = f"{dataset_name}_" if dataset_name else ""
    fig.savefig(outdir / f"{prefix}pareto_frontier.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(outdir / f"{prefix}pareto_frontier.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {outdir / f'{prefix}pareto_frontier.pdf'}")


def plot_nfr_vs_cumforgetting(df: pd.DataFrame, outdir: Path, dataset_name: str = "") -> None:
    """Generate NFR vs cumulative forgetting plot."""
    df_no_bcwi = df[df["method"] != "bcwi"]

    summary = df_no_bcwi.groupby(["method", "lam"]).agg(
        nfr_mean=("nfr", "mean"),
        nfr_std=("nfr", "std"),
        cum_forgetting_mean=("cum_forgetting", "mean"),
        cum_forgetting_std=("cum_forgetting", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))

    for method in summary["method"].unique():
        mdf = summary[summary["method"] == method].copy()
        mdf = mdf.sort_values("lam")

        color = METHOD_COLORS.get(method, "gray")
        marker = METHOD_MARKERS.get(method, "o")

        x = mdf["nfr_mean"].values
        y = mdf["cum_forgetting_mean"].values
        xerr = mdf["nfr_std"].values
        yerr = mdf["cum_forgetting_std"].values

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
    ax.set_ylabel("Cumulative Forgetting", fontsize=12)
    title = "NFR vs Cumulative Forgetting During Training"
    if dataset_name:
        title = f"{dataset_name.upper()}: {title}"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    prefix = f"{dataset_name}_" if dataset_name else ""
    fig.savefig(outdir / f"{prefix}pareto_frontier_with_cumforgetting.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(outdir / f"{prefix}pareto_frontier_with_cumforgetting.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {outdir / f'{prefix}pareto_frontier_with_cumforgetting.pdf'}")


def run_dataset_sweep(
    dataset_name: str,
    n_splits: int,
    outdir: Path,
    figdir: Path,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Run λ sweep for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    if dataset_name == "adult" and data_dir is not None:
        train_df, test_df = load_adult(data_dir)
        x_train, y_train, x_test, y_test, _ = transform_data(train_df, test_df)
    else:
        ds = load_dataset(dataset_name)
        x_train, y_train = ds.x_train, ds.y_train
        x_test, y_test = ds.x_test, ds.y_test

    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"  Class balance: {y_train.mean():.2%} positive")

    sizes = get_sample_sizes(dataset_name, len(x_train), len(x_test))
    print(f"  Sample sizes: old={sizes['old_n']}, new={sizes['new_n']}, "
          f"anchor={sizes['anchor_n']}, val={sizes['val_n']}, test={sizes['test_n']}")

    print(f"\nRunning λ sweep with {n_splits} splits...")
    raw_df = run_lambda_sweep(
        x_train,
        y_train,
        x_test,
        y_test,
        lambdas=LAMBDAS_TRAIN,
        rhos=RHOS_BCWI,
        n_splits=n_splits,
        **sizes,
    )

    raw_df["dataset"] = dataset_name

    raw_path = outdir / f"{dataset_name}_lambda_frontier_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved: {raw_path}")

    summary_df = summarize(raw_df.drop(columns=["dataset"]), ["method", "lam"])
    summary_df["dataset"] = dataset_name
    summary_path = outdir / f"{dataset_name}_lambda_frontier_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\nGenerating plots...")
    plot_pareto_frontiers(raw_df, figdir, dataset_name)
    plot_nfr_vs_cumforgetting(raw_df, figdir, dataset_name)

    return raw_df


def main():
    parser = argparse.ArgumentParser(description="λ-Frontier Sweep for Forgetting-Penalized Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=list_datasets(),
        help=f"Dataset to use (default: adult). Available: {', '.join(list_datasets())}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing adult.data and adult.test (for Adult dataset only). "
        "If not provided, downloads from UCI to ~/.cache/pareto-gd/adult/",
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

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    run_dataset_sweep(
        dataset_name=args.dataset,
        n_splits=args.n_splits,
        outdir=args.outdir,
        figdir=args.figdir,
        data_dir=args.data_dir,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
