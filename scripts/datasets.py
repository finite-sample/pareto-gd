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
    num_classes: int = 2


OPENML_DATASETS = {
    "adult": 1590,
    "bank": 1461,
    "credit": 42477,
    "diabetes": 37,
    "spambase": 44,
}

CC18_BINARY_DATASETS = {
    "adult": 1590,
    "australian": 40981,
    "bank-marketing": 1461,
    "banknote-authentication": 1462,
    "bioresponse": 4134,
    "blood-transfusion": 1464,
    "breast-w": 15,
    "churn": 40701,
    "climate-model-simulation-crashes": 40994,
    "credit-approval": 29,
    "credit-g": 31,
    "cylinder-bands": 6332,
    "diabetes": 37,
    "dresses-sales": 23381,
    "eeg-eye-state": 1471,
    "electricity": 151,
    "heart-c": 49,
    "heart-h": 50,
    "heart-statlog": 53,
    "hepatitis": 55,
    "hill-valley": 1479,
    "ilpd": 1480,
    "internet-advertisements": 40978,
    "ionosphere": 59,
    "jm1": 1053,
    "kc1": 1067,
    "kc2": 1063,
    "kr-vs-kp": 3,
    "madelon": 1485,
    "mushroom": 24,
    "nomao": 1486,
    "numerai28.6": 23517,
    "ozone-level-8hr": 1487,
    "pc1": 1068,
    "pc3": 1050,
    "pc4": 1049,
    "phishingwebsites": 4534,
    "phoneme": 1489,
    "qsar-biodeg": 1494,
    "sick": 38,
    "sonar": 40,
    "spambase": 44,
    "steel-plates-fault": 1504,
    "wdbc": 1510,
    "wilt": 40983,
}

CC18_MULTICLASS_DATASETS = {
    "analcatdata_authorship": 458,
    "analcatdata_dmft": 469,
    "anneal": 2,
    "car": 40975,
    "cmc": 23,
    "cnae-9": 1468,
    "collins": 40971,
    "connect-4": 40668,
    "dna": 40670,
    "first-order-theorem-proving": 1475,
    "gas-drift": 1476,
    "har": 1478,
    "japanesevowels": 375,
    "kdd_ipums_la_97-small": 1049,
    "ldpa": 1483,
    "letter": 6,
    "mfeat-factors": 12,
    "mfeat-fourier": 14,
    "mfeat-karhunen": 16,
    "mfeat-morphological": 18,
    "mfeat-pixel": 20,
    "mfeat-zernike": 22,
    "mnist_784": 554,
    "optdigits": 28,
    "pendigits": 32,
    "satimage": 182,
    "segment": 36,
    "semeion": 1501,
    "splice": 46,
    "texture": 40499,
    "vehicle": 54,
    "volcanoes-a1": 1527,
    "vowel": 307,
    "waveform-5000": 60,
    "wine-quality-red": 40691,
    "wine-quality-white": 40498,
    "yeast": 181,
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
    force_binary: bool = False,
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
    if force_binary and len(classes) > 2:
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
    """Load dataset by name (hardcoded or CC18)."""
    if name in DATASET_LOADERS:
        return DATASET_LOADERS[name](**kwargs)
    dataset_id = get_cc18_id(name)
    if dataset_id is not None:
        return load_cc18_dataset(dataset_id, **kwargs)
    raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())} + CC18 datasets")


def list_datasets() -> list[str]:
    """Return list of hardcoded dataset names."""
    return list(DATASET_LOADERS.keys())


def list_cc18_binary() -> list[str]:
    """Return CC18 binary classification dataset names."""
    return list(CC18_BINARY_DATASETS.keys())


def list_cc18_multiclass() -> list[str]:
    """Return CC18 multiclass classification dataset names."""
    return list(CC18_MULTICLASS_DATASETS.keys())


def list_cc18_all() -> list[str]:
    """Return all CC18 dataset names (binary + multiclass)."""
    return list_cc18_binary() + list_cc18_multiclass()


def get_cc18_id(name: str) -> int | None:
    """Get OpenML ID for a CC18 dataset name (binary or multiclass)."""
    name_lower = name.lower()
    if name_lower in CC18_BINARY_DATASETS:
        return CC18_BINARY_DATASETS[name_lower]
    if name_lower in CC18_MULTICLASS_DATASETS:
        return CC18_MULTICLASS_DATASETS[name_lower]
    return None


def is_cc18_binary(name: str) -> bool:
    """Check if a dataset is binary classification."""
    return name.lower() in CC18_BINARY_DATASETS


def load_cc18_dataset(dataset_id: int, test_size: float = 0.3, seed: int = 42, force_binary: bool = False, max_samples: int | None = None) -> BenchmarkDataset:
    """Load any CC18 dataset by OpenML ID. Optionally subsample to max_samples."""
    import openml

    train_df, test_df, target = load_openml_dataset(dataset_id, test_size, seed)

    if max_samples is not None:
        total = len(train_df) + len(test_df)
        if total > max_samples:
            frac = max_samples / total
            train_df = train_df.sample(frac=frac, random_state=seed).reset_index(drop=True)
            test_df = test_df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    x_train, y_train, x_test, y_test = preprocess_dataframe(train_df, test_df, target, force_binary=force_binary)

    ds = openml.datasets.get_dataset(dataset_id, download_data=False)
    num_classes = len(np.unique(y_train))

    return BenchmarkDataset(
        name=ds.name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_features=x_train.shape[1],
        n_samples_train=len(x_train),
        n_samples_test=len(x_test),
        num_classes=num_classes,
    )


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
