#!/usr/bin/env python3
"""
Dataset loaders for the multi-dataset λ-frontier benchmark.

Supports loading datasets from OpenML with automatic caching and preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class BenchmarkDataset:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_samples_train: int
    n_samples_test: int


OPENML_DATASETS = {
    "adult": 1590,
    "bank": 1461,
    "credit": 42477,
    "diabetes": 37,
    "spambase": 44,
}


def get_cache_dir() -> Path:
    cache = Path.home() / ".cache" / "pareto-gd"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def download_adult(cache_dir: Optional[Path] = None) -> Path:
    """Download Adult dataset from UCI if not present."""
    if cache_dir is None:
        cache_dir = get_cache_dir() / "adult"
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    for fname in ["adult.data", "adult.test"]:
        dest = cache_dir / fname
        if not dest.exists():
            print(f"Downloading {fname} from UCI...")
            urllib.request.urlretrieve(base_url + fname, dest)
    return cache_dir


def load_adult_raw(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Adult dataframes."""
    if data_dir is None:
        data_dir = get_cache_dir() / "adult"

    if not (data_dir / "adult.data").exists():
        download_adult(data_dir)

    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"
    ]

    train_df = pd.read_csv(data_dir / "adult.data", header=None, names=cols, skipinitialspace=True)
    test_df = pd.read_csv(data_dir / "adult.test", header=None, names=cols, skiprows=1, skipinitialspace=True)

    train_df["income"] = train_df["income"].astype(str).str.strip().map({"<=50K": 0, ">50K": 1})
    test_df["income"] = test_df["income"].astype(str).str.strip().map({"<=50K.": 0, ">50K.": 1})

    for c in train_df.columns:
        if train_df[c].dtype == object:
            train_df[c] = train_df[c].astype(str).str.strip().replace("?", np.nan)
            test_df[c] = test_df[c].astype(str).str.strip().replace("?", np.nan)

    return train_df, test_df


def load_openml_dataset(dataset_id: int, test_size: float = 0.3, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load dataset from OpenML and split into train/test."""
    try:
        import openml
    except ImportError:
        raise ImportError("openml package required. Install with: pip install openml")

    dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    X, y, categorical_mask, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X, columns=attribute_names)

    target_name = dataset.default_target_attribute
    df[target_name] = y

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=y)
    train_df = pd.DataFrame(train_df).reset_index(drop=True)
    test_df = pd.DataFrame(test_df).reset_index(drop=True)

    return train_df, test_df, target_name


def preprocess_dataframe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess dataframes: encode target, one-hot categoricals, scale numerics."""
    x_train_df = train_df.drop(columns=[target_col])
    x_test_df = test_df.drop(columns=[target_col])
    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.astype(str))
    y_test_enc = le.transform(y_test.astype(str))

    classes = le.classes_
    if len(classes) > 2:
        last_class_idx = list(classes).index(classes[-1])
        y_train_enc = (y_train_enc == last_class_idx).astype(int)
        y_test_enc = (y_test_enc == last_class_idx).astype(int)

    num_cols = x_train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = x_train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols
        ))

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    x_train = preprocessor.fit_transform(x_train_df).astype(np.float64)
    x_test = preprocessor.transform(x_test_df).astype(np.float64)

    return x_train, y_train_enc, x_test, y_test_enc


def load_adult(data_dir: Optional[Path] = None) -> BenchmarkDataset:
    """Load Adult dataset with preprocessing."""
    train_df, test_df = load_adult_raw(data_dir)
    x_train, y_train, x_test, y_test = preprocess_dataframe(train_df, test_df, "income")

    return BenchmarkDataset(
        name="adult",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_features=x_train.shape[1],
        n_samples_train=len(x_train),
        n_samples_test=len(x_test),
    )


def load_bank() -> BenchmarkDataset:
    """Load Bank Marketing dataset from OpenML."""
    train_df, test_df, target = load_openml_dataset(OPENML_DATASETS["bank"])
    x_train, y_train, x_test, y_test = preprocess_dataframe(train_df, test_df, target)

    return BenchmarkDataset(
        name="bank",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_features=x_train.shape[1],
        n_samples_train=len(x_train),
        n_samples_test=len(x_test),
    )


def load_credit() -> BenchmarkDataset:
    """Load Credit Default dataset from OpenML."""
    train_df, test_df, target = load_openml_dataset(OPENML_DATASETS["credit"])
    x_train, y_train, x_test, y_test = preprocess_dataframe(train_df, test_df, target)

    return BenchmarkDataset(
        name="credit",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_features=x_train.shape[1],
        n_samples_train=len(x_train),
        n_samples_test=len(x_test),
    )


def load_diabetes() -> BenchmarkDataset:
    """Load Diabetes (PIMA) dataset from OpenML."""
    train_df, test_df, target = load_openml_dataset(OPENML_DATASETS["diabetes"])
    x_train, y_train, x_test, y_test = preprocess_dataframe(train_df, test_df, target)

    return BenchmarkDataset(
        name="diabetes",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_features=x_train.shape[1],
        n_samples_train=len(x_train),
        n_samples_test=len(x_test),
    )


def load_spambase() -> BenchmarkDataset:
    """Load Spambase dataset from OpenML."""
    train_df, test_df, target = load_openml_dataset(OPENML_DATASETS["spambase"])
    x_train, y_train, x_test, y_test = preprocess_dataframe(train_df, test_df, target)

    return BenchmarkDataset(
        name="spambase",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_features=x_train.shape[1],
        n_samples_train=len(x_train),
        n_samples_test=len(x_test),
    )


DATASET_LOADERS = {
    "adult": load_adult,
    "bank": load_bank,
    "credit": load_credit,
    "diabetes": load_diabetes,
    "spambase": load_spambase,
}


def load_dataset(name: str, **kwargs) -> BenchmarkDataset:
    """Load dataset by name."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name](**kwargs)


def list_datasets() -> list[str]:
    """Return list of available dataset names."""
    return list(DATASET_LOADERS.keys())


if __name__ == "__main__":
    print("Testing dataset loaders...\n")

    for name in list_datasets():
        print(f"Loading {name}...")
        try:
            ds = load_dataset(name)
            print(f"  Train: {ds.x_train.shape}, Test: {ds.x_test.shape}")
            print(f"  Features: {ds.n_features}, Class balance: {ds.y_train.mean():.2%} positive")
        except Exception as e:
            print(f"  Error: {e}")
        print()
