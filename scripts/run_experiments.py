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
    python3 scripts/run_experiments.py --datasets adult,diabetes --n-splits 3

    # Full benchmark (all 5 datasets, 10 splits)
    python3 scripts/run_experiments.py --n-splits 10

    # Full CC18 benchmark
    python3 scripts/run_experiments.py --cc18-all --n-splits 10 --outdir results/

Requires Python 3.12+
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import BenchmarkDataset, list_cc18_all, list_cc18_binary, list_datasets, load_dataset
from models import TrainConfig, set_seed, train_erm, train_erm_multiclass
from training import (
    train_baseline,
    train_confidence_drop,
    train_fixed_anchor,
    train_selective_distill,
    train_projected_gd,
    bcwi_select,
)
from training_multiclass import (
    train_baseline_multiclass,
    train_confidence_drop_multiclass,
    train_fixed_anchor_multiclass,
    train_selective_distill_multiclass,
    train_projected_gd_multiclass,
    bcwi_select_multiclass,
)


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
    model_type: str = "mlp"


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

    incumbent = train_erm(split.x_old, split.y_old, input_dim, baseline_config, config.model_type)

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
        model_type=config.model_type,
    )
    rows.append({
        "dataset": dataset.name,
        "split": split_idx,
        "method": "baseline",
        "model_type": config.model_type,
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
            model_type=config.model_type,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "confidence_drop",
            "model_type": config.model_type,
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
            model_type=config.model_type,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "fixed_anchor",
            "model_type": config.model_type,
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
            model_type=config.model_type,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "selective_distill",
            "model_type": config.model_type,
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
            "model_type": config.model_type,
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
            model_type=config.model_type,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "bcwi",
            "model_type": config.model_type,
            "lambda": np.nan,
            "target_nfr": target_nfr,
            "achieved_nfr": bcwi_eval.nfr,
            "accuracy": bcwi_eval.accuracy,
            "pfr": bcwi_eval.pfr,
        })

    return rows


def run_split_multiclass(
    dataset: BenchmarkDataset,
    split_idx: int,
    config: ExperimentConfig,
) -> list[dict]:
    """Run all methods on a single split (multiclass version)."""
    seed = config.seed_base + split_idx
    split = create_split(dataset, seed)
    input_dim = dataset.n_features
    num_classes = dataset.num_classes

    baseline_config = TrainConfig(
        epochs=config.epochs_baseline,
        lr=config.lr,
        batch_size=config.batch_size,
        seed=seed,
    )

    incumbent = train_erm_multiclass(
        split.x_old, split.y_old, input_dim, num_classes, baseline_config, config.model_type
    )

    x_anchor_t = torch.tensor(split.x_anchor, dtype=torch.float32)
    x_val_t = torch.tensor(split.x_val, dtype=torch.float32)
    x_test_t = torch.tensor(split.x_test, dtype=torch.float32)

    incumbent_anchor_pred = incumbent.predict(x_anchor_t).numpy()
    incumbent_val_pred = incumbent.predict(x_val_t).numpy()
    incumbent_test_pred = incumbent.predict(x_test_t).numpy()

    incumbent_anchor_prob = incumbent.predict_prob(x_anchor_t).numpy()

    incumbent_anchor_correct = incumbent_anchor_pred == split.y_anchor
    x_anchor_keep = split.x_anchor[incumbent_anchor_correct]
    y_anchor_keep = split.y_anchor[incumbent_anchor_correct]
    incumbent_anchor_keep_prob = incumbent_anchor_prob[incumbent_anchor_correct]

    rows: list[dict] = []

    _, baseline_eval = train_baseline_multiclass(
        split.x_train, split.y_train,
        split.x_test, split.y_test,
        incumbent_test_pred,
        num_classes,
        TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 1),
        model_type=config.model_type,
    )
    rows.append({
        "dataset": dataset.name,
        "split": split_idx,
        "method": "baseline",
        "model_type": config.model_type,
        "lambda": np.nan,
        "target_nfr": np.nan,
        "achieved_nfr": baseline_eval.nfr,
        "accuracy": baseline_eval.accuracy,
        "pfr": baseline_eval.pfr,
    })

    for lam in config.lambda_values:
        _, cd_eval, _ = train_confidence_drop_multiclass(
            split.x_train, split.y_train,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_val_pred, incumbent_test_pred,
            num_classes,
            lam=lam,
            config=TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 2),
            warmup_epochs=config.warmup_epochs,
            model_type=config.model_type,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "confidence_drop",
            "model_type": config.model_type,
            "lambda": lam,
            "target_nfr": np.nan,
            "achieved_nfr": cd_eval.nfr,
            "accuracy": cd_eval.accuracy,
            "pfr": cd_eval.pfr,
        })

        _, fa_eval, _ = train_fixed_anchor_multiclass(
            incumbent,
            split.x_train, split.y_train,
            x_anchor_keep, y_anchor_keep,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_anchor_pred[incumbent_anchor_correct],
            incumbent_val_pred, incumbent_test_pred,
            num_classes,
            lam=lam,
            config=TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 3),
            warmup_epochs=config.warmup_epochs,
            model_type=config.model_type,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "fixed_anchor",
            "model_type": config.model_type,
            "lambda": lam,
            "target_nfr": np.nan,
            "achieved_nfr": fa_eval.nfr,
            "accuracy": fa_eval.accuracy,
            "pfr": fa_eval.pfr,
        })

        _, sd_eval, _ = train_selective_distill_multiclass(
            incumbent,
            split.x_train, split.y_train,
            x_anchor_keep,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_anchor_keep_prob,
            incumbent_val_pred, incumbent_test_pred,
            num_classes,
            lam=lam,
            config=TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 4),
            warmup_epochs=config.warmup_epochs,
            model_type=config.model_type,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "selective_distill",
            "model_type": config.model_type,
            "lambda": lam,
            "target_nfr": np.nan,
            "achieved_nfr": sd_eval.nfr,
            "accuracy": sd_eval.accuracy,
            "pfr": sd_eval.pfr,
        })

    for target_nfr in config.nfr_targets:
        _, pgd_eval, _ = train_projected_gd_multiclass(
            incumbent,
            split.x_train, split.y_train,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_val_pred, incumbent_test_pred,
            num_classes,
            target_nfr=target_nfr,
            config=TrainConfig(epochs=config.epochs_constrained, lr=config.lr, batch_size=config.batch_size, seed=seed + 5),
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "projected_gd",
            "model_type": config.model_type,
            "lambda": np.nan,
            "target_nfr": target_nfr,
            "achieved_nfr": pgd_eval.nfr,
            "accuracy": pgd_eval.accuracy,
            "pfr": pgd_eval.pfr,
        })

        _, bcwi_eval, _ = bcwi_select_multiclass(
            incumbent,
            split.x_train, split.y_train,
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            incumbent_val_pred, incumbent_test_pred,
            num_classes,
            target_nfr=target_nfr,
            config=TrainConfig(epochs=config.epochs_baseline, lr=config.lr, batch_size=config.batch_size, seed=seed + 6),
            model_type=config.model_type,
        )
        rows.append({
            "dataset": dataset.name,
            "split": split_idx,
            "method": "bcwi",
            "model_type": config.model_type,
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
    outdir: Path | None = None,
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
            if dataset.num_classes == 2:
                rows = run_split(dataset, split_idx, config)
            else:
                rows = run_split_multiclass(dataset, split_idx, config)
            all_rows.extend(rows)

        if outdir is not None:
            pd.DataFrame(all_rows).to_csv(outdir / "results_partial.csv", index=False)

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Run All Methods")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated datasets (default: all)")
    parser.add_argument("--cc18", action="store_true", help="Run on all CC18 binary classification datasets")
    parser.add_argument("--cc18-all", action="store_true", help="Run on all CC18 datasets (binary + multiclass)")
    parser.add_argument("--model", type=str, choices=["mlp", "logreg"], default="mlp", help="Model type (default: mlp)")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of splits per dataset")
    parser.add_argument("--epochs-baseline", type=int, default=50, help="Epochs for baseline methods")
    parser.add_argument("--epochs-constrained", type=int, default=100, help="Epochs for constrained methods")
    parser.add_argument("--outdir", type=Path, default=Path("tabs"), help="Output dir for results CSV")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.cc18_all:
        dataset_names = list_cc18_all()
        print(f"Running full CC18 benchmark: {len(dataset_names)} datasets (binary + multiclass)")
    elif args.cc18:
        dataset_names = list_cc18_binary()
        print(f"Running CC18 binary benchmark: {len(dataset_names)} datasets")
    elif args.datasets is not None:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    else:
        dataset_names = list_datasets()
    datasets = [load_dataset(name) for name in dataset_names]

    config = ExperimentConfig(
        n_splits=args.n_splits,
        epochs_baseline=args.epochs_baseline,
        epochs_constrained=args.epochs_constrained,
        model_type=args.model,
    )

    df = run_benchmark(datasets, config, verbose=True, outdir=args.outdir)
    df.to_csv(args.outdir / "results.csv", index=False)
    print(f"\nSaved: {args.outdir / 'results.csv'}")


if __name__ == "__main__":
    main()
