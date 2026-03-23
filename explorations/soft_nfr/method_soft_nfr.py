#!/usr/bin/env python3
"""
Soft NFR: Convex relaxation for post-hoc model selection.

Implements soft NFR as a differentiable proxy for hard NFR,
enabling gradient-based optimization over checkpoint simplex.

Requires Python 3.12+
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from metrics import EvalResult, compute_flips
from models import MLP, TrainConfig, interpolate_models, train_erm, set_seed


@dataclass(slots=True)
class SoftNFRResult:
    """Result from soft NFR optimization."""
    model: MLP
    eval_result: EvalResult
    soft_nfr_val: float
    soft_nfr_test: float
    weights: np.ndarray
    pareto_points: list[dict]


def compute_soft_nfr(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    incumbent_probs: np.ndarray,
) -> float:
    """
    Compute soft NFR = mean(max(0, model_loss - incumbent_loss)) on incumbent-correct examples.

    Args:
        model: Model to evaluate
        x: Input features
        y: True labels
        incumbent_probs: Incumbent's predicted probabilities

    Returns:
        Soft NFR value (0 = no regression, higher = more regression)
    """
    inc_pred = incumbent_probs >= 0.5
    inc_correct = inc_pred == y

    if inc_correct.sum() == 0:
        return 0.0

    x_correct = x[inc_correct]
    y_correct = y[inc_correct]
    inc_probs_correct = incumbent_probs[inc_correct]

    x_t = torch.tensor(x_correct, dtype=torch.float32)
    y_t = torch.tensor(y_correct, dtype=torch.float32)
    inc_probs_t = torch.tensor(inc_probs_correct, dtype=torch.float32)

    with torch.no_grad():
        model_logits = model(x_t)
        model_loss = F.binary_cross_entropy_with_logits(model_logits, y_t, reduction='none')

        inc_probs_clamped = torch.clamp(inc_probs_t, 1e-7, 1 - 1e-7)
        inc_loss = F.binary_cross_entropy(inc_probs_clamped, y_t, reduction='none')

        soft_nfr = torch.clamp(model_loss - inc_loss, min=0.0).mean()

    return float(soft_nfr)


def compute_soft_nfr_differentiable(
    logits: torch.Tensor,
    y: torch.Tensor,
    incumbent_loss: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable soft NFR for gradient-based optimization.

    Args:
        logits: Model logits
        y: True labels
        incumbent_loss: Pre-computed incumbent loss per example
        mask: Boolean mask for incumbent-correct examples

    Returns:
        Soft NFR tensor (differentiable)
    """
    if mask.sum() == 0:
        return torch.tensor(0.0)

    model_loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask], reduction='none')
    soft_nfr = torch.clamp(model_loss - incumbent_loss[mask], min=0.0).mean()
    return soft_nfr


def interpolate_checkpoints(
    checkpoints: list[MLP],
    weights: np.ndarray,
) -> MLP:
    """
    Create model as weighted combination of checkpoints.

    Args:
        checkpoints: List of checkpoint models
        weights: Simplex weights (sum to 1, all >= 0)

    Returns:
        Interpolated model
    """
    assert len(checkpoints) == len(weights)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-6).all()

    state_dicts = [ckpt.state_dict() for ckpt in checkpoints]
    interp_state = {}

    for key in state_dicts[0]:
        interp_state[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))

    new_model = MLP(checkpoints[0]._input_dim, checkpoints[0]._hidden_dims)
    new_model.load_state_dict(interp_state)
    return new_model


def soft_nfr_1d(
    incumbent: MLP,
    candidate: MLP,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_val_prob: np.ndarray,
    incumbent_test_prob: np.ndarray,
    n_alphas: int = 101,
) -> SoftNFRResult:
    """
    Post-hoc optimization over 1D line (incumbent ↔ candidate).

    This is a fair comparison to BCWI but using soft NFR instead of hard NFR.
    Returns full Pareto frontier data instead of selecting a single point.

    Args:
        incumbent: Incumbent model
        candidate: Candidate model
        x_val, y_val: Validation data
        x_test, y_test: Test data
        incumbent_val_prob: Incumbent probabilities on validation set
        incumbent_test_prob: Incumbent probabilities on test set
        n_alphas: Number of alpha values to search

    Returns:
        SoftNFRResult with best model and Pareto frontier data
    """
    alphas = np.linspace(0.0, 1.0, n_alphas)
    pareto_points: list[dict] = []

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

        pareto_points.append({
            'alpha': alpha,
            'weights': np.array([alpha, 1 - alpha]),
            'val_acc': val_acc,
            'val_hard_nfr': val_flips.nfr,
            'val_soft_nfr': val_soft_nfr,
            'test_acc': test_acc,
            'test_hard_nfr': test_flips.nfr,
            'test_soft_nfr': test_soft_nfr,
            'test_pfr': test_flips.pfr,
        })

    best_idx = max(range(len(pareto_points)), key=lambda i: pareto_points[i]['val_acc'])
    best_point = pareto_points[best_idx]
    best_model = interpolate_models(incumbent, candidate, best_point['alpha'])

    eval_result = EvalResult(
        accuracy=best_point['test_acc'],
        nfr=best_point['test_hard_nfr'],
        pfr=best_point['test_pfr'],
        prob=best_model.predict_prob(x_test_t).numpy(),
    )

    return SoftNFRResult(
        model=best_model,
        eval_result=eval_result,
        soft_nfr_val=best_point['val_soft_nfr'],
        soft_nfr_test=best_point['test_soft_nfr'],
        weights=best_point['weights'],
        pareto_points=pareto_points,
    )


def soft_nfr_kd(
    checkpoints: list[MLP],
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_val_prob: np.ndarray,
    incumbent_test_prob: np.ndarray,
    n_grid: int = 21,
    use_gradient: bool = True,
    lr: float = 0.1,
    n_iters: int = 100,
) -> SoftNFRResult:
    """
    Post-hoc optimization over k-dimensional checkpoint simplex.

    For small k, uses grid search. For larger k, uses gradient descent
    with softmax reparameterization.

    Args:
        checkpoints: List of k checkpoint models (first is incumbent)
        x_val, y_val: Validation data
        x_test, y_test: Test data
        incumbent_val_prob: Incumbent probabilities on validation set
        incumbent_test_prob: Incumbent probabilities on test set
        n_grid: Grid points per dimension (for grid search)
        use_gradient: Whether to use gradient descent (vs pure grid)
        lr: Learning rate for gradient descent
        n_iters: Number of gradient descent iterations

    Returns:
        SoftNFRResult with best model and Pareto frontier data
    """
    k = len(checkpoints)
    pareto_points: list[dict] = []

    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    def evaluate_weights(weights: np.ndarray) -> dict:
        model = interpolate_checkpoints(checkpoints, weights)

        val_prob = model.predict_prob(x_val_t).numpy()
        val_flips = compute_flips(val_prob, incumbent_val_prob, y_val)
        val_acc = float(((val_prob >= 0.5).astype(int) == y_val).mean())
        val_soft_nfr = compute_soft_nfr(model, x_val, y_val, incumbent_val_prob)

        test_prob = model.predict_prob(x_test_t).numpy()
        test_flips = compute_flips(test_prob, incumbent_test_prob, y_test)
        test_acc = float(((test_prob >= 0.5).astype(int) == y_test).mean())
        test_soft_nfr = compute_soft_nfr(model, x_test, y_test, incumbent_test_prob)

        return {
            'weights': weights.copy(),
            'val_acc': val_acc,
            'val_hard_nfr': val_flips.nfr,
            'val_soft_nfr': val_soft_nfr,
            'test_acc': test_acc,
            'test_hard_nfr': test_flips.nfr,
            'test_soft_nfr': test_soft_nfr,
            'test_pfr': test_flips.pfr,
        }

    if k == 2:
        alphas = np.linspace(0.0, 1.0, n_grid)
        for alpha in alphas:
            weights = np.array([alpha, 1 - alpha])
            pareto_points.append(evaluate_weights(weights))

    elif k == 3:
        for i in range(n_grid):
            for j in range(n_grid - i):
                w0 = i / (n_grid - 1)
                w1 = j / (n_grid - 1)
                w2 = 1 - w0 - w1
                if w2 >= -1e-6:
                    weights = np.array([max(0, w0), max(0, w1), max(0, w2)])
                    weights = weights / weights.sum()
                    pareto_points.append(evaluate_weights(weights))

    else:
        n_samples = min(n_grid ** min(k, 3), 1000)
        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            raw = rng.exponential(1.0, k)
            weights = raw / raw.sum()
            pareto_points.append(evaluate_weights(weights))

    if use_gradient and k > 2:
        best_grid = max(pareto_points, key=lambda p: p['val_acc'])
        z = torch.tensor(np.log(best_grid['weights'] + 1e-8), requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr)

        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        for _ in range(n_iters):
            optimizer.zero_grad()

            weights = F.softmax(z, dim=0)

            state_dicts = [ckpt.state_dict() for ckpt in checkpoints]
            interp_state = {}
            for key in state_dicts[0]:
                interp_state[key] = sum(
                    weights[i] * state_dicts[i][key].clone()
                    for i in range(k)
                )

            temp_model = MLP(checkpoints[0]._input_dim, checkpoints[0]._hidden_dims)
            temp_model.load_state_dict(interp_state)

            logits = temp_model(x_val_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_val_t)

            loss.backward()
            optimizer.step()

        final_weights = F.softmax(z, dim=0).detach().numpy()
        pareto_points.append(evaluate_weights(final_weights))

    best_idx = max(range(len(pareto_points)), key=lambda i: pareto_points[i]['val_acc'])
    best_point = pareto_points[best_idx]
    best_model = interpolate_checkpoints(checkpoints, best_point['weights'])

    eval_result = EvalResult(
        accuracy=best_point['test_acc'],
        nfr=best_point['test_hard_nfr'],
        pfr=best_point['test_pfr'],
        prob=best_model.predict_prob(x_test_t).numpy(),
    )

    return SoftNFRResult(
        model=best_model,
        eval_result=eval_result,
        soft_nfr_val=best_point['val_soft_nfr'],
        soft_nfr_test=best_point['test_soft_nfr'],
        weights=best_point['weights'],
        pareto_points=pareto_points,
    )


def train_with_checkpoints(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    config: TrainConfig,
    checkpoint_every: int = 10,
) -> list[MLP]:
    """
    Train model and save checkpoints at regular intervals.

    Args:
        x_train, y_train: Training data
        input_dim: Input dimension
        config: Training configuration
        checkpoint_every: Save checkpoint every N epochs

    Returns:
        List of checkpoint models
    """
    set_seed(config.seed)

    model = MLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    checkpoints: list[MLP] = []
    n = len(x_train)

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

        if (epoch + 1) % checkpoint_every == 0:
            checkpoints.append(model.clone())

    if not checkpoints or (config.epochs % checkpoint_every != 0):
        checkpoints.append(model.clone())

    return checkpoints


def soft_nfr_posthoc(
    incumbent: MLP,
    checkpoints: list[MLP],
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    incumbent_val_prob: np.ndarray,
    incumbent_test_prob: np.ndarray,
    n_alphas: int = 101,
    use_gradient: bool = True,
) -> SoftNFRResult:
    """
    Post-hoc optimization using soft NFR.

    If checkpoints has 1 element (candidate), uses 1D search.
    If checkpoints has multiple elements, searches over simplex.
    Incumbent is always included as the first vertex of the simplex.

    Args:
        incumbent: Incumbent model
        checkpoints: List of candidate checkpoint(s)
        x_val, y_val: Validation data
        x_test, y_test: Test data
        incumbent_val_prob: Incumbent probabilities on validation set
        incumbent_test_prob: Incumbent probabilities on test set
        n_alphas: Number of points for grid search
        use_gradient: Whether to use gradient descent for kD

    Returns:
        SoftNFRResult with model, evaluation, and Pareto frontier
    """
    if len(checkpoints) == 1:
        return soft_nfr_1d(
            incumbent=incumbent,
            candidate=checkpoints[0],
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            incumbent_val_prob=incumbent_val_prob,
            incumbent_test_prob=incumbent_test_prob,
            n_alphas=n_alphas,
        )
    else:
        all_checkpoints = [incumbent] + checkpoints
        return soft_nfr_kd(
            checkpoints=all_checkpoints,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            incumbent_val_prob=incumbent_val_prob,
            incumbent_test_prob=incumbent_test_prob,
            n_grid=n_alphas,
            use_gradient=use_gradient,
        )


if __name__ == "__main__":
    from datasets import load_dataset

    print("Testing soft NFR methods on diabetes dataset...")
    ds = load_dataset("diabetes")

    input_dim = ds.n_features
    config = TrainConfig(epochs=50, seed=42)

    incumbent = train_erm(ds.x_train[:200], ds.y_train[:200], input_dim, config)
    candidate = train_erm(ds.x_train, ds.y_train, input_dim, TrainConfig(epochs=50, seed=43))

    x_val_t = torch.tensor(ds.x_test[:100], dtype=torch.float32)
    x_test_t = torch.tensor(ds.x_test[100:], dtype=torch.float32)
    incumbent_val_prob = incumbent.predict_prob(x_val_t).numpy()
    incumbent_test_prob = incumbent.predict_prob(x_test_t).numpy()

    print("\n1D Soft NFR (fair BCWI comparison):")
    result_1d = soft_nfr_1d(
        incumbent=incumbent,
        candidate=candidate,
        x_val=ds.x_test[:100],
        y_val=ds.y_test[:100],
        x_test=ds.x_test[100:],
        y_test=ds.y_test[100:],
        incumbent_val_prob=incumbent_val_prob,
        incumbent_test_prob=incumbent_test_prob,
    )
    print(f"  Test Acc: {result_1d.eval_result.accuracy:.4f}")
    print(f"  Test Hard NFR: {result_1d.eval_result.nfr:.4f}")
    print(f"  Test Soft NFR: {result_1d.soft_nfr_test:.4f}")
    print(f"  Pareto points: {len(result_1d.pareto_points)}")

    print("\nkD Soft NFR (with checkpoints):")
    checkpoints = train_with_checkpoints(
        ds.x_train, ds.y_train, input_dim,
        TrainConfig(epochs=50, seed=43),
        checkpoint_every=10,
    )
    print(f"  Checkpoints saved: {len(checkpoints)}")

    result_kd = soft_nfr_posthoc(
        incumbent=incumbent,
        checkpoints=checkpoints,
        x_val=ds.x_test[:100],
        y_val=ds.y_test[:100],
        x_test=ds.x_test[100:],
        y_test=ds.y_test[100:],
        incumbent_val_prob=incumbent_val_prob,
        incumbent_test_prob=incumbent_test_prob,
    )
    print(f"  Test Acc: {result_kd.eval_result.accuracy:.4f}")
    print(f"  Test Hard NFR: {result_kd.eval_result.nfr:.4f}")
    print(f"  Test Soft NFR: {result_kd.soft_nfr_test:.4f}")
    print(f"  Pareto points: {len(result_kd.pareto_points)}")
    print(f"  Best weights shape: {result_kd.weights.shape}")
