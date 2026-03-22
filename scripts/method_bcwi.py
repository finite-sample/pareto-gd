#!/usr/bin/env python3
"""
BCWI: Backwards Compatible Weight Interpolation.

Train candidate freely via ERM, then interpolate with incumbent:
θ_interp = α·θ_incumbent + (1-α)·θ_candidate

Select α that achieves NFR ≤ target with best accuracy.

This method is restricted to the 1D line segment between incumbent and candidate.

Requires Python 3.12+
"""

import numpy as np
import torch

from metrics import EvalResult, compute_flips, evaluate
from models import MLP, TrainConfig, interpolate_models, train_erm


def bcwi(
    incumbent: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_val_prob: np.ndarray,
    incumbent_test_prob: np.ndarray,
    target_nfr: float,
    config: TrainConfig,
    n_alphas: int = 101,
) -> tuple[MLP, EvalResult, dict]:
    """
    Run BCWI method.

    Returns: (model, eval_result, extra_info)
    """
    input_dim = x_train.shape[1]
    candidate = train_erm(x_train, y_train, input_dim, config)

    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    alphas = np.linspace(0.0, 1.0, n_alphas)

    best_model: MLP | None = None
    best_alpha: float = 0.0
    best_acc: float = -1.0
    best_nfr: float = float("inf")

    for alpha in alphas:
        model = interpolate_models(incumbent, candidate, alpha)
        prob = model.predict_prob(x_val_t).numpy()
        flips = compute_flips(prob, incumbent_val_prob, y_val)
        acc = float(((prob >= 0.5).astype(int) == y_val).mean())

        if flips.nfr <= target_nfr + 0.001 and acc > best_acc:
            best_acc = acc
            best_nfr = flips.nfr
            best_alpha = alpha
            best_model = model

    if best_model is None:
        for alpha in alphas:
            model = interpolate_models(incumbent, candidate, alpha)
            prob = model.predict_prob(x_val_t).numpy()
            flips = compute_flips(prob, incumbent_val_prob, y_val)

            if flips.nfr < best_nfr:
                best_nfr = flips.nfr
                best_acc = float(((prob >= 0.5).astype(int) == y_val).mean())
                best_alpha = alpha
                best_model = model

    assert best_model is not None
    eval_result = evaluate(best_model, x_test, y_test, incumbent_test_prob)

    return best_model, eval_result, {"alpha": best_alpha}
