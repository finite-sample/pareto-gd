#!/usr/bin/env python3
"""
Analyze benchmark results: Pareto frontiers, correlations, win rates.

Usage:
    python3 explorations/soft_nfr/analyze.py
    python3 explorations/soft_nfr/analyze.py --input results/pareto_frontier.csv

Requires Python 3.12+
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, minimize_x: bool = True, minimize_y: bool = True) -> pd.DataFrame:
    """
    Extract Pareto-optimal points from a dataframe.

    Args:
        df: DataFrame with columns x_col and y_col
        x_col: Column for x-axis (e.g., 'test_hard_nfr')
        y_col: Column for y-axis (e.g., 'test_acc')
        minimize_x: Whether to minimize x (True for NFR)
        minimize_y: Whether to minimize y (False for accuracy)

    Returns:
        DataFrame with only Pareto-optimal points
    """
    points = df[[x_col, y_col]].values
    is_pareto = np.ones(len(points), dtype=bool)

    for i, (x, y) in enumerate(points):
        for j, (ox, oy) in enumerate(points):
            if i != j:
                x_better = (ox < x) if minimize_x else (ox > x)
                y_better = (oy < y) if minimize_y else (oy > y)
                x_equal = np.isclose(ox, x)
                y_equal = np.isclose(oy, y)

                if (x_better and (y_better or y_equal)) or (y_better and (x_better or x_equal)):
                    is_pareto[i] = False
                    break

    return df[is_pareto].sort_values(x_col)


def compute_auc_pareto(df: pd.DataFrame, x_col: str = 'test_hard_nfr', y_col: str = 'test_acc') -> float:
    """
    Compute area under Pareto frontier (higher is better).

    Uses trapezoidal integration of accuracy over NFR range [0, max_nfr].
    """
    pareto = compute_pareto_frontier(df, x_col, y_col, minimize_x=True, minimize_y=False)
    if len(pareto) < 2:
        return pareto[y_col].iloc[0] if len(pareto) == 1 else 0.0

    x = pareto[x_col].values
    y = pareto[y_col].values

    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    y = y[sorted_idx]

    auc = np.trapezoid(y, x)

    if x[-1] > x[0]:
        auc = auc / (x[-1] - x[0])

    return float(auc)


def compute_soft_hard_correlation(df: pd.DataFrame) -> dict:
    """Compute correlation between soft NFR and hard NFR."""
    valid = df.dropna(subset=['test_soft_nfr', 'test_hard_nfr'])
    if len(valid) < 3:
        return {'pearson_r': np.nan, 'spearman_r': np.nan, 'p_value': np.nan}

    pearson_r, pearson_p = stats.pearsonr(valid['test_soft_nfr'], valid['test_hard_nfr'])
    spearman_r, spearman_p = stats.spearmanr(valid['test_soft_nfr'], valid['test_hard_nfr'])

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
    }


def compute_win_rates(df: pd.DataFrame, nfr_thresholds: list[float]) -> pd.DataFrame:
    """
    Compute accuracy win rates at specific NFR thresholds.

    For each threshold, find best accuracy at NFR <= threshold for each method.
    """
    rows: list[dict] = []

    for dataset in df['dataset'].unique():
        for split in df[df['dataset'] == dataset]['split'].unique():
            split_df = df[(df['dataset'] == dataset) & (df['split'] == split)]

            for threshold in nfr_thresholds:
                method_accs = {}

                for method in split_df['method'].unique():
                    method_df = split_df[split_df['method'] == method]
                    feasible = method_df[method_df['test_hard_nfr'] <= threshold + 0.001]

                    if len(feasible) > 0:
                        method_accs[method] = feasible['test_acc'].max()
                    else:
                        closest = method_df.loc[method_df['test_hard_nfr'].idxmin()]
                        method_accs[method] = closest['test_acc']

                if method_accs:
                    best_acc = max(method_accs.values())
                    winners = [m for m, a in method_accs.items() if np.isclose(a, best_acc, atol=0.001)]

                    rows.append({
                        'dataset': dataset,
                        'split': split,
                        'nfr_threshold': threshold,
                        **{f'{m}_acc': method_accs.get(m, np.nan) for m in ['bcwi', 'soft_nfr_1d', 'soft_nfr_kd']},
                        'winner': winners[0] if len(winners) == 1 else 'tie',
                    })

    return pd.DataFrame(rows)


def plot_pareto_single_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    outdir: Path,
    split: int = 0,
) -> None:
    """Plot Pareto frontiers for a single dataset and split."""
    ddf = df[(df['dataset'] == dataset_name) & (df['split'] == split)]

    colors = {
        'bcwi': '#9467bd',
        'soft_nfr_1d': '#2ca02c',
        'soft_nfr_kd': '#d62728',
    }
    labels = {
        'bcwi': 'BCWI (hard NFR)',
        'soft_nfr_1d': 'Soft-NFR-1D',
        'soft_nfr_kd': 'Soft-NFR-kD',
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in ['bcwi', 'soft_nfr_1d', 'soft_nfr_kd']:
        mdf = ddf[ddf['method'] == method]
        if mdf.empty:
            continue

        pareto = compute_pareto_frontier(mdf, 'test_hard_nfr', 'test_acc', minimize_x=True, minimize_y=False)

        ax.scatter(
            mdf['test_hard_nfr'],
            mdf['test_acc'],
            c=colors[method],
            alpha=0.2,
            s=20,
        )

        ax.plot(
            pareto['test_hard_nfr'],
            pareto['test_acc'],
            '-o',
            color=colors[method],
            label=labels[method],
            markersize=6,
            linewidth=2,
        )

    ax.set_xlabel('Hard NFR (Negative Flip Rate)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'{dataset_name}: Pareto Frontiers (Split {split})', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{dataset_name}_split{split}_pareto.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_pareto_all_datasets(df: pd.DataFrame, outdir: Path) -> None:
    """Plot combined faceted Pareto frontiers for all datasets."""
    datasets = df['dataset'].unique()
    n = len(datasets)

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    colors = {
        'bcwi': '#9467bd',
        'soft_nfr_1d': '#2ca02c',
        'soft_nfr_kd': '#d62728',
    }

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.atleast_2d(axes).flatten()

    for idx, dataset_name in enumerate(datasets):
        ax = axes_flat[idx]
        ddf = df[df['dataset'] == dataset_name]

        summary = ddf.groupby(['method', 'split']).apply(
            lambda g: pd.Series({
                'nfr_at_best_acc': g.loc[g['test_acc'].idxmax(), 'test_hard_nfr'],
                'best_acc': g['test_acc'].max(),
            }),
            include_groups=False,
        ).reset_index()

        mean_summary = summary.groupby('method').agg({
            'nfr_at_best_acc': 'mean',
            'best_acc': 'mean',
        }).reset_index()

        for method in ['bcwi', 'soft_nfr_1d', 'soft_nfr_kd']:
            mrow = mean_summary[mean_summary['method'] == method]
            if mrow.empty:
                continue
            ax.scatter(
                mrow['nfr_at_best_acc'],
                mrow['best_acc'],
                marker='o' if method == 'bcwi' else ('s' if method == 'soft_nfr_1d' else '^'),
                c=colors[method],
                s=100,
                zorder=5,
            )

        ax.set_xlabel('NFR', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(dataset_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['bcwi'], markersize=10, label='BCWI'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['soft_nfr_1d'], markersize=10, label='Soft-NFR-1D'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['soft_nfr_kd'], markersize=10, label='Soft-NFR-kD'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'all_datasets_pareto.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_soft_vs_hard_correlation(df: pd.DataFrame, outdir: Path) -> None:
    """Plot soft NFR vs hard NFR correlation."""
    fig, ax = plt.subplots(figsize=(8, 6))

    valid = df.dropna(subset=['test_soft_nfr', 'test_hard_nfr'])

    ax.scatter(
        valid['test_soft_nfr'],
        valid['test_hard_nfr'],
        alpha=0.3,
        s=10,
    )

    corr = compute_soft_hard_correlation(df)
    ax.set_xlabel('Soft NFR', fontsize=12)
    ax.set_ylabel('Hard NFR', fontsize=12)
    ax.set_title(f'Soft vs Hard NFR Correlation (r={corr["pearson_r"]:.3f})', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'soft_vs_hard_correlation.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)


def print_summary(df: pd.DataFrame, win_rates: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("Soft NFR Benchmark Summary")
    print("=" * 70)

    corr = compute_soft_hard_correlation(df)
    print(f"\nSoft vs Hard NFR Correlation:")
    print(f"  Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
    print(f"  Spearman r: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    print("\nAUC under Pareto Frontier by Method:")
    for dataset in df['dataset'].unique():
        print(f"\n  {dataset}:")
        for method in ['bcwi', 'soft_nfr_1d', 'soft_nfr_kd']:
            mdf = df[(df['dataset'] == dataset) & (df['method'] == method)]
            if mdf.empty:
                continue
            aucs = []
            for split in mdf['split'].unique():
                split_df = mdf[mdf['split'] == split]
                auc = compute_auc_pareto(split_df)
                aucs.append(auc)
            print(f"    {method:15s}: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

    print("\nWin Rates by NFR Threshold:")
    for threshold in win_rates['nfr_threshold'].unique():
        tdf = win_rates[win_rates['nfr_threshold'] == threshold]
        total = len(tdf)
        print(f"\n  NFR <= {threshold:.1%}:")
        for method in ['bcwi', 'soft_nfr_1d', 'soft_nfr_kd', 'tie']:
            wins = (tdf['winner'] == method).sum()
            print(f"    {method:15s}: {wins}/{total} ({100*wins/total:.1f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze Soft NFR Benchmark Results")
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).parent / "results" / "pareto_frontier.csv",
                        help="Input CSV file")
    parser.add_argument("--outdir", type=Path,
                        default=Path(__file__).parent / "results",
                        help="Output directory for plots")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run benchmark.py first to generate results.")
        return

    print(f"Loading results from {args.input}")
    df = pd.read_csv(args.input)

    args.outdir.mkdir(parents=True, exist_ok=True)

    for dataset in df['dataset'].unique():
        plot_pareto_single_dataset(df, dataset, args.outdir, split=0)
        print(f"Saved: {args.outdir / f'{dataset}_split0_pareto.pdf'}")

    plot_pareto_all_datasets(df, args.outdir)
    print(f"Saved: {args.outdir / 'all_datasets_pareto.pdf'}")

    plot_soft_vs_hard_correlation(df, args.outdir)
    print(f"Saved: {args.outdir / 'soft_vs_hard_correlation.pdf'}")

    nfr_thresholds = [0.0, 0.005, 0.01, 0.02, 0.03]
    win_rates = compute_win_rates(df, nfr_thresholds)
    win_rates.to_csv(args.outdir / 'win_rates.csv', index=False)
    print(f"Saved: {args.outdir / 'win_rates.csv'}")

    print_summary(df, win_rates)


if __name__ == "__main__":
    main()
