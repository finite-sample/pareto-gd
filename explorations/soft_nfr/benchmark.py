#!/usr/bin/env python3
"""
Benchmark: Compare BCWI, Soft-NFR-1D, and Soft-NFR-kD methods.

Traces full Pareto frontiers (accuracy vs hard NFR) for each method.
No single-point selection - sweep and record all trade-off points.

Usage:
    python3 explorations/soft_nfr/benchmark.py --datasets adult,diabetes --n-splits 3
    python3 explorations/soft_nfr/benchmark.py --n-splits 10

Requires Python 3.12+
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from datasets import BenchmarkDataset, list_datasets, load_dataset, load_cc18_dataset, list_cc18_all as list_cc18
from metrics import compute_flips
from models import MLP, TrainConfig, interpolate_models, set_seed, train_erm
from method_soft_nfr import (
    compute_soft_nfr,
    soft_nfr_1d,
    soft_nfr_posthoc,
    train_with_checkpoints,
)


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for benchmark."""
    n_splits: int = 10
    epochs_incumbent: int = 50
    epochs_candidate: int = 50
    lr: float = 0.001
    batch_size: int = 256
    seed_base: int = 200
    n_alphas: int = 101
    checkpoint_every: int = 10


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

    return SplitData(
        x_old=dataset.x_train[old_idx],
        y_old=dataset.y_train[old_idx],
        x_train=dataset.x_train[train_perm],
        y_train=dataset.y_train[train_perm],
        x_val=dataset.x_test[val_idx],
        y_val=dataset.y_test[val_idx],
        x_test=dataset.x_test[test_idx],
        y_test=dataset.y_test[test_idx],
    )


def trace_bcwi_frontier(
    incumbent: MLP,
    candidate: MLP,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_val_prob: np.ndarray,
    incumbent_test_prob: np.ndarray,
    n_alphas: int = 101,
) -> list[dict]:
    """
    Trace BCWI Pareto frontier using hard NFR.

    Returns list of dicts with alpha, accuracy, hard_nfr, soft_nfr for each point.
    """
    alphas = np.linspace(0.0, 1.0, n_alphas)
    points: list[dict] = []

    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    for alpha in alphas:
        model = interpolate_models(incumbent, candidate, alpha)

        val_prob = model.predict_prob(x_val_t).numpy()
        val_flips = compute_flips(val_prob, incumbent_val_prob, y_val)
        val_acc = float(((val_prob >= 0.5).astype(int) == y_val).mean())
        val_soft_nfr = compute_soft_nfr(model, x_val, y_val, incumbent_val_prob)

        test_prob = model.predict_prob(x_test_t).numpy()
        test_flips = compute_flips(test_prob, incumbent_test_prob, y_test)
        test_acc = float(((test_prob >= 0.5).astype(int) == y_test).mean())
        test_soft_nfr = compute_soft_nfr(model, x_test, y_test, incumbent_test_prob)

        points.append({
            'alpha': alpha,
            'val_acc': val_acc,
            'val_hard_nfr': val_flips.nfr,
            'val_soft_nfr': val_soft_nfr,
            'test_acc': test_acc,
            'test_hard_nfr': test_flips.nfr,
            'test_soft_nfr': test_soft_nfr,
            'test_pfr': test_flips.pfr,
        })

    return points


def run_split(
    dataset: BenchmarkDataset,
    split_idx: int,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    """Run all methods on a single split, return Pareto frontier data."""
    seed = config.seed_base + split_idx
    split = create_split(dataset, seed)
    input_dim = dataset.n_features

    incumbent_config = TrainConfig(
        epochs=config.epochs_incumbent,
        lr=config.lr,
        batch_size=config.batch_size,
        seed=seed,
    )
    candidate_config = TrainConfig(
        epochs=config.epochs_candidate,
        lr=config.lr,
        batch_size=config.batch_size,
        seed=seed + 1,
    )

    incumbent = train_erm(split.x_old, split.y_old, input_dim, incumbent_config)

    x_val_t = torch.tensor(split.x_val, dtype=torch.float32)
    x_test_t = torch.tensor(split.x_test, dtype=torch.float32)
    incumbent_val_prob = incumbent.predict_prob(x_val_t).numpy()
    incumbent_test_prob = incumbent.predict_prob(x_test_t).numpy()

    candidate = train_erm(split.x_train, split.y_train, input_dim, candidate_config)

    rows: list[dict] = []

    bcwi_points = trace_bcwi_frontier(
        incumbent=incumbent,
        candidate=candidate,
        x_val=split.x_val,
        y_val=split.y_val,
        x_test=split.x_test,
        y_test=split.y_test,
        incumbent_val_prob=incumbent_val_prob,
        incumbent_test_prob=incumbent_test_prob,
        n_alphas=config.n_alphas,
    )

    for pt in bcwi_points:
        rows.append({
            'dataset': dataset.name,
            'split': split_idx,
            'method': 'bcwi',
            **pt,
        })

    soft_nfr_1d_result = soft_nfr_1d(
        incumbent=incumbent,
        candidate=candidate,
        x_val=split.x_val,
        y_val=split.y_val,
        x_test=split.x_test,
        y_test=split.y_test,
        incumbent_val_prob=incumbent_val_prob,
        incumbent_test_prob=incumbent_test_prob,
        n_alphas=config.n_alphas,
    )

    for pt in soft_nfr_1d_result.pareto_points:
        rows.append({
            'dataset': dataset.name,
            'split': split_idx,
            'method': 'soft_nfr_1d',
            **pt,
        })

    checkpoints = train_with_checkpoints(
        split.x_train, split.y_train, input_dim,
        TrainConfig(
            epochs=config.epochs_candidate,
            lr=config.lr,
            batch_size=config.batch_size,
            seed=seed + 1,
        ),
        checkpoint_every=config.checkpoint_every,
    )

    soft_nfr_kd_result = soft_nfr_posthoc(
        incumbent=incumbent,
        checkpoints=checkpoints,
        x_val=split.x_val,
        y_val=split.y_val,
        x_test=split.x_test,
        y_test=split.y_test,
        incumbent_val_prob=incumbent_val_prob,
        incumbent_test_prob=incumbent_test_prob,
        n_alphas=21,
        use_gradient=True,
    )

    for pt in soft_nfr_kd_result.pareto_points:
        rows.append({
            'dataset': dataset.name,
            'split': split_idx,
            'method': 'soft_nfr_kd',
            'n_checkpoints': len(checkpoints) + 1,
            **pt,
        })

    return pd.DataFrame(rows)


def run_benchmark(
    datasets: list[BenchmarkDataset],
    config: BenchmarkConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run benchmark across all datasets and splits."""
    all_dfs: list[pd.DataFrame] = []

    for dataset in datasets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset.name}")
            print(f"  Train: {dataset.x_train.shape}, Test: {dataset.x_test.shape}")
            print(f"{'='*60}")

        for split_idx in range(config.n_splits):
            if verbose:
                print(f"  Split {split_idx + 1}/{config.n_splits}...")
            df = run_split(dataset, split_idx, config)
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Soft NFR Benchmark")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all hardcoded)")
    parser.add_argument("--cc18", action="store_true",
                        help="Run on all 72 CC18 datasets (multi-class converted to binary)")
    parser.add_argument("--dataset-ids", type=str, default=None,
                        help="Comma-separated OpenML dataset IDs")
    parser.add_argument("--n-splits", type=int, default=10,
                        help="Number of splits per dataset")
    parser.add_argument("--epochs-incumbent", type=int, default=50,
                        help="Epochs for incumbent training")
    parser.add_argument("--epochs-candidate", type=int, default=50,
                        help="Epochs for candidate training")
    parser.add_argument("--n-alphas", type=int, default=101,
                        help="Number of alpha values for 1D methods")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--outdir", type=Path,
                        default=Path(__file__).parent / "results",
                        help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample large datasets to this size")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    datasets: list[BenchmarkDataset] = []

    if args.cc18:
        cc18_ids = list_cc18()
        print(f"Loading {len(cc18_ids)} CC18 datasets (multi-class converted to binary)...")
        for dataset_id in cc18_ids:
            try:
                ds = load_cc18_dataset(dataset_id, max_samples=args.max_samples)
                datasets.append(ds)
                print(f"  Loaded: {ds.name} (id={dataset_id}, n={ds.n_samples_train + ds.n_samples_test})")
            except Exception as e:
                print(f"  SKIP: dataset {dataset_id} failed: {e}")
    elif args.dataset_ids:
        ids = [int(d.strip()) for d in args.dataset_ids.split(",")]
        for dataset_id in ids:
            try:
                ds = load_cc18_dataset(dataset_id, max_samples=args.max_samples)
                datasets.append(ds)
                print(f"  Loaded: {ds.name} (id={dataset_id}, n={ds.n_samples_train + ds.n_samples_test})")
            except Exception as e:
                print(f"  SKIP: dataset {dataset_id} failed: {e}")
    else:
        dataset_names = (
            list_datasets() if args.datasets is None
            else [d.strip() for d in args.datasets.split(",")]
        )
        datasets = [load_dataset(name) for name in dataset_names]

    config = BenchmarkConfig(
        n_splits=args.n_splits,
        epochs_incumbent=args.epochs_incumbent,
        epochs_candidate=args.epochs_candidate,
        n_alphas=args.n_alphas,
        checkpoint_every=args.checkpoint_every,
    )

    df = run_benchmark(datasets, config, verbose=True)

    outfile = args.outdir / "pareto_frontier.csv"
    df.to_csv(outfile, index=False)
    print(f"\nSaved: {outfile}")

    print("\nSummary by method:")
    summary = df.groupby('method').agg({
        'test_acc': ['mean', 'std'],
        'test_hard_nfr': ['mean', 'std'],
        'test_soft_nfr': ['mean', 'std'],
    }).round(4)
    print(summary)


if __name__ == "__main__":
    main()
