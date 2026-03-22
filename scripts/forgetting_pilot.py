#!/usr/bin/env python3
"""
Compact pilot for incumbent-vs-update regression testing on Adult.

What it does
------------
1. Loads Adult from:
   - adult.data
   - adult.test

2. Builds a compact benchmark with:
   - incumbent trained on an "old" subset
   - candidate trained on old + new
   - evaluation on final negative flips vs incumbent
   - cumulative forgetting on a held-out golden set
   - policy-change stress test on a chosen slice

3. Compares:
   - baseline ERM
   - confidence_drop (epoch-local loss increase penalty)
   - fixed_anchor (fixed-incumbent loss anchor)
   - selective_distill (distill to incumbent on anchor set)
   - bcwi (post-hoc weight interpolation between incumbent and candidate)

Outputs
-------
- stable_split_results.csv
- stable_summary.csv
- policy_results.csv
- summary.json

This is a compact pilot script, not an exact reproduction of the paper's
precise Adult MLP experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.neural_network import MLPClassifier


ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
NUM_COLS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bce_from_prob(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace("?", np.nan)
    return df


def download_adult(cache_dir: Path) -> Path:
    """Download Adult dataset from UCI if not present."""
    import urllib.request
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    for fname in ["adult.data", "adult.test"]:
        dest = cache_dir / fname
        if not dest.exists():
            print(f"Downloading {fname} from UCI...")
            urllib.request.urlretrieve(base_url + fname, dest)
            print(f"  Saved to {dest}")
    return cache_dir


def load_adult(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data_dir is None:
        data_dir = Path.home() / ".cache" / "pareto-gd" / "adult"

    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"

    if not train_path.exists() or not test_path.exists():
        download_adult(data_dir)

    train_df = pd.read_csv(train_path, header=None, names=ADULT_COLS, skipinitialspace=True)
    test_df = pd.read_csv(test_path, header=None, names=ADULT_COLS, skiprows=1, skipinitialspace=True)

    train_df["income"] = train_df["income"].astype(str).str.strip().map({"<=50K": 0, ">50K": 1})
    test_df["income"] = test_df["income"].astype(str).str.strip().map({"<=50K.": 0, ">50K.": 1})

    train_df = clean_df(train_df)
    test_df = clean_df(test_df)
    return train_df, test_df


def build_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    x_df = train_df.drop(columns=["income"])
    cat_cols = [c for c in x_df.columns if c not in NUM_COLS]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), NUM_COLS),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        sparse_threshold=0.0,
    )


def transform_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    prep = build_preprocessor(train_df)
    x_train = prep.fit_transform(train_df.drop(columns=["income"])).astype(np.float64)
    x_test = prep.transform(test_df.drop(columns=["income"])).astype(np.float64)
    y_train = train_df["income"].astype(int).to_numpy()
    y_test = test_df["income"].astype(int).to_numpy()
    return x_train, y_train, x_test, y_test, prep


def predict_prob(clf: MLPClassifier, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(x)[:, 1]


def clone_mlp_params(clf: MLPClassifier) -> List[np.ndarray]:
    return [w.copy() for w in clf.coefs_] + [b.copy() for b in clf.intercepts_]


def set_mlp_params(clf: MLPClassifier, params: List[np.ndarray]) -> None:
    n_w = len(clf.coefs_)
    for i in range(n_w):
        clf.coefs_[i] = params[i].copy()
    for i in range(len(clf.intercepts_)):
        clf.intercepts_[i] = params[n_w + i].copy()


def interpolate_params(p_old: List[np.ndarray], p_new: List[np.ndarray], alpha: float) -> List[np.ndarray]:
    return [alpha * a + (1 - alpha) * b for a, b in zip(p_old, p_new)]


@dataclass
class Metrics:
    acc: float
    nfr: float
    nfr_count: int
    nfr_ref_count: int
    pos_flip: float
    pos_flip_count: int
    pos_flip_ref_count: int
    cum_forgetting: int


def compute_flip_metrics(
    incumbent_prob: np.ndarray,
    candidate_prob: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, int, int, float, int, int]:
    inc_pred = (incumbent_prob >= 0.5)
    cand_pred = (candidate_prob >= 0.5)
    inc_correct = (inc_pred == y)
    cand_correct = (cand_pred == y)

    nfr_mask = inc_correct & (~cand_correct)
    pos_mask = (~inc_correct) & cand_correct

    nfr_ref = int(inc_correct.sum())
    pos_ref = int((~inc_correct).sum())

    nfr_count = int(nfr_mask.sum())
    pos_count = int(pos_mask.sum())

    nfr = nfr_count / nfr_ref if nfr_ref else float("nan")
    pos_flip = pos_count / pos_ref if pos_ref else float("nan")
    return nfr, nfr_count, nfr_ref, pos_flip, pos_count, pos_ref


def make_base_clf(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=1,
        shuffle=True,
        random_state=seed,
        warm_start=True,
        tol=0.0,
        n_iter_no_change=200,
    )


def fit_baseline_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_golden: np.ndarray,
    y_golden: np.ndarray,
    epochs: int,
    seed: int,
    warmup: int = 10,
):
    clf = make_base_clf(seed)
    classes = np.array([0, 1])

    prev_correct = None
    cum_forgetting = 0
    epoch_probs_golden = []

    for epoch in range(epochs):
        if epoch == 0:
            clf.partial_fit(x_train, y_train, classes=classes)
        else:
            clf.partial_fit(x_train, y_train)

        pg = predict_prob(clf, x_golden)
        epoch_probs_golden.append(pg.copy())
        cur_correct = ((pg >= 0.5).astype(int) == y_golden)
        if prev_correct is not None and epoch >= warmup:
            cum_forgetting += int((prev_correct & (~cur_correct)).sum())
        prev_correct = cur_correct

    return clf, cum_forgetting, epoch_probs_golden


def fit_confidence_drop(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_golden: np.ndarray,
    y_golden: np.ndarray,
    lam: float,
    epochs: int,
    seed: int,
    warmup: int = 10,
    replay_frac: float = 0.25,
):
    """
    Practical approximation:
    - fit one epoch on original labels
    - after warmup, identify high loss-increase examples relative to previous epoch
      and replay them with oversampling
    """
    clf = make_base_clf(seed)
    classes = np.array([0, 1])

    prev_loss = None
    prev_correct = None
    cum_forgetting = 0

    x_cur = x_train
    y_cur = y_train

    for epoch in range(epochs):
        if epoch == 0:
            clf.partial_fit(x_cur, y_cur, classes=classes)
        else:
            clf.partial_fit(x_cur, y_cur)

        train_prob = predict_prob(clf, x_train)
        cur_loss = bce_from_prob(train_prob, y_train)

        if prev_loss is not None and epoch >= warmup:
            delta = np.maximum(cur_loss - prev_loss, 0.0)
            if delta.sum() > 0:
                n_replay = max(1, int(replay_frac * len(x_train)))
                w = delta / delta.sum()
                idx = np.random.choice(len(x_train), size=n_replay, replace=True, p=w)
                x_cur = np.vstack([x_train, x_train[idx]])
                y_cur = np.concatenate([y_train, y_train[idx]])
            else:
                x_cur = x_train
                y_cur = y_train
        else:
            x_cur = x_train
            y_cur = y_train

        prev_loss = cur_loss

        pg = predict_prob(clf, x_golden)
        cur_correct = ((pg >= 0.5).astype(int) == y_golden)
        if prev_correct is not None and epoch >= warmup:
            cum_forgetting += int((prev_correct & (~cur_correct)).sum())
        prev_correct = cur_correct

    return clf, cum_forgetting


def fit_fixed_anchor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_anchor: np.ndarray,
    y_anchor: np.ndarray,
    incumbent_anchor_prob: np.ndarray,
    x_golden: np.ndarray,
    y_golden: np.ndarray,
    lam: float,
    epochs: int,
    seed: int,
    warmup: int = 10,
    replay_frac: float = 0.25,
):
    """
    Practical approximation:
    after warmup, oversample anchor examples where candidate loss exceeds incumbent loss.
    """
    clf = make_base_clf(seed)
    classes = np.array([0, 1])

    incumbent_anchor_loss = bce_from_prob(incumbent_anchor_prob, y_anchor)
    prev_correct = None
    cum_forgetting = 0

    x_cur = x_train
    y_cur = y_train

    for epoch in range(epochs):
        if epoch == 0:
            clf.partial_fit(x_cur, y_cur, classes=classes)
        else:
            clf.partial_fit(x_cur, y_cur)

        if epoch >= warmup:
            anchor_prob = predict_prob(clf, x_anchor)
            cur_anchor_loss = bce_from_prob(anchor_prob, y_anchor)
            penalty = np.maximum(cur_anchor_loss - incumbent_anchor_loss, 0.0)
            if penalty.sum() > 0:
                n_replay = max(1, int(replay_frac * len(x_anchor)))
                w = penalty / penalty.sum()
                idx = np.random.choice(len(x_anchor), size=n_replay, replace=True, p=w)
                k = max(1, int(round(lam)))
                x_aug = [x_train] + [x_anchor[idx]] * k
                y_aug = [y_train] + [y_anchor[idx]] * k
                x_cur = np.vstack(x_aug)
                y_cur = np.concatenate(y_aug)
            else:
                x_cur = x_train
                y_cur = y_train
        else:
            x_cur = x_train
            y_cur = y_train

        pg = predict_prob(clf, x_golden)
        cur_correct = ((pg >= 0.5).astype(int) == y_golden)
        if prev_correct is not None and epoch >= warmup:
            cum_forgetting += int((prev_correct & (~cur_correct)).sum())
        prev_correct = cur_correct

    return clf, cum_forgetting


def fit_selective_distill(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_anchor: np.ndarray,
    incumbent_anchor_prob: np.ndarray,
    x_golden: np.ndarray,
    y_golden: np.ndarray,
    lam: float,
    epochs: int,
    seed: int,
    warmup: int = 10,
    replay_frac: float = 0.25,
):
    """
    Practical approximation:
    build soft labels on anchor points from incumbent and replay them after warmup.
    """
    clf = make_base_clf(seed)
    classes = np.array([0, 1])

    soft_anchor = (incumbent_anchor_prob >= 0.5).astype(int)

    prev_correct = None
    cum_forgetting = 0
    x_cur = x_train
    y_cur = y_train

    for epoch in range(epochs):
        if epoch == 0:
            clf.partial_fit(x_cur, y_cur, classes=classes)
        else:
            clf.partial_fit(x_cur, y_cur)

        if epoch >= warmup:
            n_replay = max(1, int(replay_frac * len(x_anchor)))
            idx = np.random.choice(len(x_anchor), size=n_replay, replace=True)
            k = max(1, int(round(lam)))
            x_aug = [x_train] + [x_anchor[idx]] * k
            y_aug = [y_train] + [soft_anchor[idx]] * k
            x_cur = np.vstack(x_aug)
            y_cur = np.concatenate(y_aug)
        else:
            x_cur = x_train
            y_cur = y_train

        pg = predict_prob(clf, x_golden)
        cur_correct = ((pg >= 0.5).astype(int) == y_golden)
        if prev_correct is not None and epoch >= warmup:
            cum_forgetting += int((prev_correct & (~cur_correct)).sum())
        prev_correct = cur_correct

    return clf, cum_forgetting


def bcwi_select(
    incumbent_clf: MLPClassifier,
    candidate_clf: MLPClassifier,
    x_val: np.ndarray,
    y_val: np.ndarray,
    incumbent_val_prob: np.ndarray,
    rho: float = 2.0,
    alphas: Optional[List[float]] = None,
) -> Tuple[MLPClassifier, float]:
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]

    p_old = clone_mlp_params(incumbent_clf)
    p_new = clone_mlp_params(candidate_clf)

    best_alpha = None
    best_score = None
    best_clf = None

    idx_0 = np.where(y_val == 0)[0][:5]
    idx_1 = np.where(y_val == 1)[0][:5]
    init_idx = np.concatenate([idx_0, idx_1])
    x_init, y_init = x_val[init_idx], y_val[init_idx]

    for alpha in alphas:
        clf = make_base_clf(seed=0)
        clf.partial_fit(x_init, y_init, classes=np.array([0, 1]))
        set_mlp_params(clf, interpolate_params(p_old, p_new, alpha))
        val_prob = predict_prob(clf, x_val)
        val_err = 1.0 - accuracy_score(y_val, (val_prob >= 0.5).astype(int))
        nfr, *_ = compute_flip_metrics(incumbent_val_prob, val_prob, y_val)
        score = val_err + rho * nfr
        if best_score is None or score < best_score:
            best_score = score
            best_alpha = alpha
            best_clf = clf

    return best_clf, float(best_alpha)


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in group_cols + ["split", "seed", "alpha"]]
    out_rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=1) if len(sub) > 1 else 0.0)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def evaluate_candidate(
    candidate_prob: np.ndarray,
    incumbent_prob: np.ndarray,
    y: np.ndarray,
    cum_forgetting: int,
) -> Metrics:
    acc = accuracy_score(y, (candidate_prob >= 0.5).astype(int))
    nfr, nfr_count, nfr_ref, pos_flip, pos_count, pos_ref = compute_flip_metrics(
        incumbent_prob, candidate_prob, y
    )
    return Metrics(
        acc=float(acc),
        nfr=float(nfr),
        nfr_count=int(nfr_count),
        nfr_ref_count=int(nfr_ref),
        pos_flip=float(pos_flip),
        pos_flip_count=int(pos_count),
        pos_flip_ref_count=int(pos_ref),
        cum_forgetting=int(cum_forgetting),
    )


def run_stable_pilot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    outdir: Path,
    n_splits: int = 2,
    old_n: int = 5000,
    new_n: int = 8000,
    anchor_n: int = 1500,
    val_n: int = 2500,
    test_n: int = 6000,
):
    rows = []

    for split in range(n_splits):
        seed = 100 + split
        set_seed(seed)

        train_idx = np.arange(len(x_train))
        test_idx = np.arange(len(x_test))

        train_sub = np.random.choice(train_idx, size=old_n + new_n, replace=False)
        old_idx = train_sub[:old_n]
        new_idx = train_sub[old_n:]

        test_sub = np.random.choice(test_idx, size=anchor_n + val_n + test_n, replace=False)
        anchor_idx = test_sub[:anchor_n]
        val_idx = test_sub[anchor_n:anchor_n + val_n]
        test_eval_idx = test_sub[anchor_n + val_n:]

        x_old, y_old = x_train[old_idx], y_train[old_idx]
        x_new, y_new = x_train[new_idx], y_train[new_idx]
        x_cand = np.vstack([x_old, x_new])
        y_cand = np.concatenate([y_old, y_new])

        x_anchor, y_anchor = x_test[anchor_idx], y_test[anchor_idx]
        x_val, y_val = x_test[val_idx], y_test[val_idx]
        x_eval, y_eval = x_test[test_eval_idx], y_test[test_eval_idx]

        incumbent, _, _ = fit_baseline_epochs(x_old, y_old, x_anchor, y_anchor, epochs=20, seed=seed)
        incumbent_anchor_prob = predict_prob(incumbent, x_anchor)
        incumbent_val_prob = predict_prob(incumbent, x_val)
        incumbent_eval_prob = predict_prob(incumbent, x_eval)

        incumbent_anchor_correct = ((incumbent_anchor_prob >= 0.5).astype(int) == y_anchor)
        x_anchor_keep = x_anchor[incumbent_anchor_correct]
        y_anchor_keep = y_anchor[incumbent_anchor_correct]
        incumbent_anchor_keep_prob = incumbent_anchor_prob[incumbent_anchor_correct]

        # Baseline
        cand_base, cf_base, _ = fit_baseline_epochs(x_cand, y_cand, x_anchor, y_anchor, epochs=20, seed=seed + 1)
        base_eval = evaluate_candidate(
            predict_prob(cand_base, x_eval), incumbent_eval_prob, y_eval, cf_base
        )
        rows.append({
            "split": split,
            "method": "baseline",
            **asdict(base_eval),
        })

        # Confidence Drop
        cand_cd, cf_cd = fit_confidence_drop(
            x_cand, y_cand, x_anchor, y_anchor, lam=1.0, epochs=20, seed=seed + 2
        )
        cd_eval = evaluate_candidate(
            predict_prob(cand_cd, x_eval), incumbent_eval_prob, y_eval, cf_cd
        )
        rows.append({
            "split": split,
            "method": "confidence_drop",
            **asdict(cd_eval),
        })

        # Fixed Anchor
        cand_fa, cf_fa = fit_fixed_anchor(
            x_cand, y_cand,
            x_anchor_keep, y_anchor_keep, incumbent_anchor_keep_prob,
            x_anchor, y_anchor,
            lam=1.0, epochs=20, seed=seed + 3
        )
        fa_eval = evaluate_candidate(
            predict_prob(cand_fa, x_eval), incumbent_eval_prob, y_eval, cf_fa
        )
        rows.append({
            "split": split,
            "method": "fixed_anchor",
            **asdict(fa_eval),
        })

        # Selective Distill
        cand_sd, cf_sd = fit_selective_distill(
            x_cand, y_cand,
            x_anchor_keep, incumbent_anchor_keep_prob,
            x_anchor, y_anchor,
            lam=1.0, epochs=20, seed=seed + 4
        )
        sd_eval = evaluate_candidate(
            predict_prob(cand_sd, x_eval), incumbent_eval_prob, y_eval, cf_sd
        )
        rows.append({
            "split": split,
            "method": "selective_distill",
            **asdict(sd_eval),
        })

        # BCWI on top of baseline candidate
        bcwi_clf, alpha = bcwi_select(incumbent, cand_base, x_val, y_val, incumbent_val_prob, rho=2.0)
        bcwi_eval = evaluate_candidate(
            predict_prob(bcwi_clf, x_eval), incumbent_eval_prob, y_eval, cum_forgetting=0
        )
        rows.append({
            "split": split,
            "method": "bcwi",
            "alpha": alpha,
            **asdict(bcwi_eval),
        })

    stable_df = pd.DataFrame(rows)
    stable_df.to_csv(outdir / "stable_split_results.csv", index=False)
    stable_summary = summarize(stable_df, ["method"])
    stable_summary.to_csv(outdir / "stable_summary.csv", index=False)
    return stable_df, stable_summary


def run_policy_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    outdir: Path,
):
    """
    Compact stress test:
    slice = female & hours_per_week > 40
    new policy labels that slice as positive
    """
    slice_train = ((train_df["sex"] == "Female") & (train_df["hours_per_week"] > 40)).to_numpy()
    slice_test = ((test_df["sex"] == "Female") & (test_df["hours_per_week"] > 40)).to_numpy()

    y_train_new = y_train.copy()
    y_test_new = y_test.copy()
    y_train_new[slice_train] = 1
    y_test_new[slice_test] = 1

    seed = 321
    set_seed(seed)

    sub_train = np.random.choice(np.arange(len(x_train)), size=13000, replace=False)
    old_idx = sub_train[:5000]
    new_idx = sub_train[5000:]

    sub_test = np.random.choice(np.arange(len(x_test)), size=7500, replace=False)
    anchor_idx = sub_test[:1500]
    eval_idx = sub_test[1500:]

    x_old, y_old = x_train[old_idx], y_train[old_idx]
    x_new, y_new = x_train[new_idx], y_train_new[new_idx]
    x_cand = np.vstack([x_old, x_new])
    y_cand = np.concatenate([y_old, y_new])

    x_anchor = x_test[anchor_idx]
    y_anchor_old = y_test[anchor_idx]
    x_eval = x_test[eval_idx]
    y_eval_new = y_test_new[eval_idx]

    incumbent, _, _ = fit_baseline_epochs(x_old, y_old, x_anchor, y_anchor_old, epochs=20, seed=seed)
    incumbent_anchor_prob = predict_prob(incumbent, x_anchor)
    incumbent_eval_prob = predict_prob(incumbent, x_eval)

    anchor_mask = ((incumbent_anchor_prob >= 0.5).astype(int) == y_anchor_old)
    anchor_keep_idx = anchor_idx[anchor_mask]

    selective_keep = anchor_keep_idx[~slice_test[anchor_keep_idx]]
    x_anchor_sel = x_test[selective_keep]
    y_anchor_sel = y_test[selective_keep]
    incumbent_anchor_sel_prob = predict_prob(incumbent, x_anchor_sel)

    changed_mask_eval = slice_test[eval_idx]

    rows = []

    cand_base, cf_base, _ = fit_baseline_epochs(x_cand, y_cand, x_anchor, y_anchor_old, epochs=20, seed=seed + 1)
    base_prob = predict_prob(cand_base, x_eval)
    rows.append({
        "method": "baseline",
        "overall_acc": float(accuracy_score(y_eval_new, (base_prob >= 0.5).astype(int))),
        "changed_slice_acc": float(accuracy_score(y_eval_new[changed_mask_eval], (base_prob[changed_mask_eval] >= 0.5).astype(int))),
    })

    cand_cd, _ = fit_confidence_drop(x_cand, y_cand, x_anchor, y_anchor_old, lam=1.0, epochs=20, seed=seed + 2)
    cd_prob = predict_prob(cand_cd, x_eval)
    rows.append({
        "method": "confidence_drop",
        "overall_acc": float(accuracy_score(y_eval_new, (cd_prob >= 0.5).astype(int))),
        "changed_slice_acc": float(accuracy_score(y_eval_new[changed_mask_eval], (cd_prob[changed_mask_eval] >= 0.5).astype(int))),
    })

    x_anchor_global = x_test[anchor_keep_idx]
    y_anchor_global = y_test[anchor_keep_idx]
    incumbent_anchor_global_prob = predict_prob(incumbent, x_anchor_global)

    cand_fa, _ = fit_fixed_anchor(
        x_cand, y_cand,
        x_anchor_global, y_anchor_global, incumbent_anchor_global_prob,
        x_anchor, y_anchor_old,
        lam=1.0, epochs=20, seed=seed + 3
    )
    fa_prob = predict_prob(cand_fa, x_eval)
    rows.append({
        "method": "fixed_anchor",
        "overall_acc": float(accuracy_score(y_eval_new, (fa_prob >= 0.5).astype(int))),
        "changed_slice_acc": float(accuracy_score(y_eval_new[changed_mask_eval], (fa_prob[changed_mask_eval] >= 0.5).astype(int))),
    })

    cand_sd, _ = fit_selective_distill(
        x_cand, y_cand,
        x_anchor_global, incumbent_anchor_global_prob,
        x_anchor, y_anchor_old,
        lam=1.0, epochs=20, seed=seed + 4
    )
    sd_prob = predict_prob(cand_sd, x_eval)
    rows.append({
        "method": "selective_distill",
        "overall_acc": float(accuracy_score(y_eval_new, (sd_prob >= 0.5).astype(int))),
        "changed_slice_acc": float(accuracy_score(y_eval_new[changed_mask_eval], (sd_prob[changed_mask_eval] >= 0.5).astype(int))),
    })

    cand_fas, _ = fit_fixed_anchor(
        x_cand, y_cand,
        x_anchor_sel, y_anchor_sel, incumbent_anchor_sel_prob,
        x_anchor, y_anchor_old,
        lam=1.0, epochs=20, seed=seed + 5
    )
    fas_prob = predict_prob(cand_fas, x_eval)
    rows.append({
        "method": "fixed_anchor_selective",
        "overall_acc": float(accuracy_score(y_eval_new, (fas_prob >= 0.5).astype(int))),
        "changed_slice_acc": float(accuracy_score(y_eval_new[changed_mask_eval], (fas_prob[changed_mask_eval] >= 0.5).astype(int))),
    })

    policy_df = pd.DataFrame(rows)
    policy_df.to_csv(outdir / "policy_results.csv", index=False)
    return policy_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing adult.data and adult.test. "
                             "If not provided, downloads from UCI to ~/.cache/pareto-gd/adult/")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_adult(args.data_dir)
    x_train, y_train, x_test, y_test, _ = transform_data(train_df, test_df)

    stable_df, stable_summary = run_stable_pilot(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, outdir=args.outdir
    )
    policy_df = run_policy_test(
        train_df=train_df, test_df=test_df,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        outdir=args.outdir,
    )

    summary = {
        "stable_methods": stable_summary.to_dict(orient="records"),
        "policy_methods": policy_df.to_dict(orient="records"),
        "notes": {
            "benchmark": "compact pilot",
            "model": "sklearn MLPClassifier",
            "stable_splits": 2,
            "objective": "final negative flips vs incumbent, plus cumulative forgetting diagnostic",
        },
    }

    with open(args.outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote results to: {args.outdir}")


if __name__ == "__main__":
    main()
