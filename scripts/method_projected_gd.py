#!/usr/bin/env python3
"""
Projected Gradient Descent for NFR-constrained optimization.

1. Initialize from incumbent
2. Take gradient step toward ERM objective
3. Project back to feasible region (NFR ≤ ε) by interpolating with incumbent

This explores beyond the 1D line segment that BCWI is restricted to.

Requires Python 3.12+
"""

import numpy as np
import torch
import torch.nn.functional as F

from metrics import EvalResult, compute_flips, evaluate
from models import MLP, TrainConfig, interpolate_models, set_seed


def _project_to_feasible(
    model: MLP,
    incumbent: MLP,
    x_val: np.ndarray,
    y_val: np.ndarray,
    incumbent_prob: np.ndarray,
    target_nfr: float,
) -> float:
    """Binary search for smallest α such that NFR ≤ target."""
    x_t = torch.tensor(x_val, dtype=torch.float32)

    cur_prob = model.predict_prob(x_t).numpy()
    cur_flips = compute_flips(cur_prob, incumbent_prob, y_val)

    if cur_flips.nfr <= target_nfr:
        return 0.0

    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        interp = interpolate_models(incumbent, model, mid)
        prob = interp.predict_prob(x_t).numpy()
        flips = compute_flips(prob, incumbent_prob, y_val)

        if flips.nfr <= target_nfr:
            hi = mid
        else:
            lo = mid

    interp = interpolate_models(incumbent, model, hi)
    model.load_state_dict(interp.state_dict())
    return hi


def projected_gd(
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
    project_every: int = 1,
) -> tuple[MLP, EvalResult, dict]:
    """
    Run projected gradient descent method.

    Returns: (model, eval_result, extra_info)
    """
    set_seed(config.seed)

    model = incumbent.clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    n = len(x_train)
    final_alpha = 0.0

    for epoch in range(config.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb, yb = x_t[idx], y_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % project_every == 0:
            final_alpha = _project_to_feasible(
                model, incumbent, x_val, y_val, incumbent_val_prob, target_nfr
            )

    eval_result = evaluate(model, x_test, y_test, incumbent_test_prob)

    return model, eval_result, {"final_alpha": final_alpha}
