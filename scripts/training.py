#!/usr/bin/env python3
"""
Training functions for NFR-constrained optimization.

This module provides training approaches for preventing model regression:
1. train_baseline - Standard ERM training
2. train_confidence_drop - Penalizes any per-example loss increase
3. train_fixed_anchor - Uses incumbent loss as anchor on fixed set
4. train_selective_distill - Distills to incumbent predictions
5. train_projected_gd - Projected gradient descent with NFR constraint
6. bcwi_select - Post-hoc weight interpolation

Requires Python 3.12+
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import EvalResult, FlipMetrics, compute_flips, evaluate
from models import MLP, TrainConfig, interpolate_models, set_seed, train_erm


def train_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_test_prob: np.ndarray,
    config: TrainConfig,
) -> tuple[MLP, EvalResult]:
    """
    Train model with standard ERM (no NFR constraint).

    Returns: (model, eval_result)
    """
    input_dim = x_train.shape[1]
    model = train_erm(x_train, y_train, input_dim, config)
    eval_result = evaluate(model, x_test, y_test, incumbent_test_prob)
    return model, eval_result


def train_confidence_drop(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_val_prob: np.ndarray,
    incumbent_test_prob: np.ndarray,
    lam: float,
    config: TrainConfig,
    warmup_epochs: int = 10,
) -> tuple[MLP, EvalResult, dict]:
    """
    Train with confidence drop penalty.

    Penalizes any increase in per-example loss compared to previous epoch.
    This implements a "do no harm" principle at the loss level.

    Returns: (model, eval_result, extra_info)
    """
    set_seed(config.seed)
    input_dim = x_train.shape[1]
    model = MLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    n = len(x_train)
    prev_loss_per_example: torch.Tensor | None = None
    total_penalty = 0.0

    for epoch in range(config.epochs):
        perm = torch.randperm(n)

        with torch.no_grad():
            logits_all = model(x_t)
            cur_loss_per_example = F.binary_cross_entropy_with_logits(
                logits_all, y_t, reduction="none"
            )

        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb, yb = x_t[idx], y_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            base_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if epoch >= warmup_epochs and prev_loss_per_example is not None:
                with torch.no_grad():
                    prev_batch = prev_loss_per_example[idx]
                cur_batch = F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
                penalty = torch.clamp(cur_batch - prev_batch, min=0.0).mean()
                loss = base_loss + lam * penalty
                total_penalty += float(penalty.detach())
            else:
                loss = base_loss

            loss.backward()
            optimizer.step()

        prev_loss_per_example = cur_loss_per_example.detach()

    eval_result = evaluate(model, x_test, y_test, incumbent_test_prob)
    return model, eval_result, {"total_penalty": total_penalty}


def train_fixed_anchor(
    incumbent: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_anchor: np.ndarray,
    y_anchor: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_anchor_prob: np.ndarray,
    incumbent_val_prob: np.ndarray,
    incumbent_test_prob: np.ndarray,
    lam: float,
    config: TrainConfig,
    warmup_epochs: int = 10,
) -> tuple[MLP, EvalResult, dict]:
    """
    Train with fixed anchor penalty.

    Penalizes when candidate loss on anchor set exceeds incumbent loss.
    The anchor set is fixed and typically consists of examples the incumbent
    classified correctly.

    Returns: (model, eval_result, extra_info)
    """
    set_seed(config.seed)
    input_dim = x_train.shape[1]
    model = MLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    x_anchor_t = torch.tensor(x_anchor, dtype=torch.float32)
    y_anchor_t = torch.tensor(y_anchor, dtype=torch.float32)

    with torch.no_grad():
        incumbent_anchor_loss = F.binary_cross_entropy_with_logits(
            torch.logit(torch.tensor(incumbent_anchor_prob, dtype=torch.float32)),
            y_anchor_t,
            reduction="none",
        )

    n = len(x_train)
    total_penalty = 0.0

    for epoch in range(config.epochs):
        perm = torch.randperm(n)

        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb, yb = x_t[idx], y_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            base_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if epoch >= warmup_epochs:
                anchor_logits = model(x_anchor_t)
                anchor_loss = F.binary_cross_entropy_with_logits(
                    anchor_logits, y_anchor_t, reduction="none"
                )
                penalty = torch.clamp(anchor_loss - incumbent_anchor_loss, min=0.0).mean()
                loss = base_loss + lam * penalty
                total_penalty += float(penalty.detach())
            else:
                loss = base_loss

            loss.backward()
            optimizer.step()

    eval_result = evaluate(model, x_test, y_test, incumbent_test_prob)
    return model, eval_result, {"total_penalty": total_penalty}


def train_selective_distill(
    incumbent: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_anchor: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_anchor_prob: np.ndarray,
    incumbent_val_prob: np.ndarray,
    incumbent_test_prob: np.ndarray,
    lam: float,
    config: TrainConfig,
    warmup_epochs: int = 10,
) -> tuple[MLP, EvalResult, dict]:
    """
    Train with selective distillation.

    Distills to incumbent predictions on the anchor set (examples where
    incumbent was correct). This preserves incumbent behavior on those
    examples while allowing learning on new data.

    Returns: (model, eval_result, extra_info)
    """
    set_seed(config.seed)
    input_dim = x_train.shape[1]
    model = MLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    x_anchor_t = torch.tensor(x_anchor, dtype=torch.float32)
    incumbent_anchor_t = torch.tensor(incumbent_anchor_prob, dtype=torch.float32)

    n = len(x_train)
    total_distill_loss = 0.0

    for epoch in range(config.epochs):
        perm = torch.randperm(n)

        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb, yb = x_t[idx], y_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            base_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if epoch >= warmup_epochs:
                anchor_probs = torch.sigmoid(model(x_anchor_t))
                distill_loss = F.mse_loss(anchor_probs, incumbent_anchor_t)
                loss = base_loss + lam * distill_loss
                total_distill_loss += float(distill_loss.detach())
            else:
                loss = base_loss

            loss.backward()
            optimizer.step()

    eval_result = evaluate(model, x_test, y_test, incumbent_test_prob)
    return model, eval_result, {"total_distill_loss": total_distill_loss}


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


def train_projected_gd(
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
    Train with projected gradient descent.

    1. Initialize from incumbent
    2. Take gradient step toward ERM objective
    3. Project back to feasible region (NFR ≤ ε) by interpolating with incumbent

    This explores beyond the 1D line segment that BCWI is restricted to.

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


def bcwi_select(
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

    Train candidate freely via ERM, then interpolate with incumbent:
    θ_interp = α·θ_incumbent + (1-α)·θ_candidate

    Select α that achieves NFR ≤ target with best accuracy.
    This method is restricted to the 1D line segment between incumbent and candidate.

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
