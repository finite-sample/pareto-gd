#!/usr/bin/env python3
"""
Metrics for NFR experiments.

Supports both binary and multiclass classification.

Requires Python 3.12+
"""

from dataclasses import dataclass

import numpy as np
import torch

from models import MLP, LogReg, MLPMulticlass, LogRegMulticlass


@dataclass(slots=True)
class FlipMetrics:
    """Flip rate metrics."""
    nfr: float
    pfr: float
    nfr_count: int
    nfr_denom: int
    pfr_count: int
    pfr_denom: int


@dataclass(slots=True)
class EvalResult:
    """Complete evaluation result for a model."""
    accuracy: float
    nfr: float
    pfr: float
    prob: np.ndarray


def compute_flips(
    candidate_prob: np.ndarray,
    incumbent_prob: np.ndarray,
    y: np.ndarray,
) -> FlipMetrics:
    """
    Compute flip rates for binary classification.

    NFR = (# flips from correct→wrong) / (# incumbent was correct)
    PFR = (# flips from wrong→correct) / (# incumbent was wrong)
    """
    inc_pred = incumbent_prob >= 0.5
    cand_pred = candidate_prob >= 0.5
    inc_correct = inc_pred == y
    cand_correct = cand_pred == y

    nfr_mask = inc_correct & (~cand_correct)
    nfr_denom = int(inc_correct.sum())
    nfr_count = int(nfr_mask.sum())
    nfr = nfr_count / nfr_denom if nfr_denom > 0 else 0.0

    pfr_mask = (~inc_correct) & cand_correct
    pfr_denom = int((~inc_correct).sum())
    pfr_count = int(pfr_mask.sum())
    pfr = pfr_count / pfr_denom if pfr_denom > 0 else 0.0

    return FlipMetrics(
        nfr=nfr, pfr=pfr,
        nfr_count=nfr_count, nfr_denom=nfr_denom,
        pfr_count=pfr_count, pfr_denom=pfr_denom,
    )


def compute_flips_multiclass(
    candidate_pred: np.ndarray,
    incumbent_pred: np.ndarray,
    y: np.ndarray,
) -> FlipMetrics:
    """
    Compute flip rates for multiclass classification.

    NFR = (# flips from correct→wrong) / (# incumbent was correct)
    PFR = (# flips from wrong→correct) / (# incumbent was wrong)
    """
    inc_correct = incumbent_pred == y
    cand_correct = candidate_pred == y

    nfr_mask = inc_correct & (~cand_correct)
    nfr_denom = int(inc_correct.sum())
    nfr_count = int(nfr_mask.sum())
    nfr = nfr_count / nfr_denom if nfr_denom > 0 else 0.0

    pfr_mask = (~inc_correct) & cand_correct
    pfr_denom = int((~inc_correct).sum())
    pfr_count = int(pfr_mask.sum())
    pfr = pfr_count / pfr_denom if pfr_denom > 0 else 0.0

    return FlipMetrics(
        nfr=nfr, pfr=pfr,
        nfr_count=nfr_count, nfr_denom=nfr_denom,
        pfr_count=pfr_count, pfr_denom=pfr_denom,
    )


def evaluate(
    model: MLP | LogReg,
    x: np.ndarray,
    y: np.ndarray,
    incumbent_prob: np.ndarray,
) -> EvalResult:
    """Evaluate binary model against incumbent."""
    x_t = torch.tensor(x, dtype=torch.float32)
    prob = model.predict_prob(x_t).numpy()
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == y).mean())
    flips = compute_flips(prob, incumbent_prob, y)

    return EvalResult(accuracy=acc, nfr=flips.nfr, pfr=flips.pfr, prob=prob)


def evaluate_multiclass(
    model: MLPMulticlass | LogRegMulticlass,
    x: np.ndarray,
    y: np.ndarray,
    incumbent_pred: np.ndarray,
) -> EvalResult:
    """Evaluate multiclass model against incumbent."""
    x_t = torch.tensor(x, dtype=torch.float32)
    prob = model.predict_prob(x_t).numpy()
    pred = prob.argmax(axis=1)
    acc = float((pred == y).mean())
    flips = compute_flips_multiclass(pred, incumbent_pred, y)

    return EvalResult(accuracy=acc, nfr=flips.nfr, pfr=flips.pfr, prob=prob)
