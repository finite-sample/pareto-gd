#!/usr/bin/env python3
"""
PyTorch MLP model and training utilities.

Requires Python 3.12+
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """Simple MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self._input_dim = input_dim
        self._hidden_dims = hidden_dims

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def clone(self) -> Self:
        """Create a deep copy of this model."""
        new_model = MLP(self._input_dim, self._hidden_dims)
        new_model.load_state_dict({k: v.clone() for k, v in self.state_dict().items()})
        return new_model


def interpolate_models(model_a: MLP, model_b: MLP, alpha: float) -> MLP:
    """Create interpolated model: alpha * model_a + (1 - alpha) * model_b."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    interp_state = {k: alpha * state_a[k] + (1 - alpha) * state_b[k] for k in state_a}

    new_model = MLP(model_a._input_dim, model_a._hidden_dims)
    new_model.load_state_dict(interp_state)
    return new_model


@dataclass(slots=True)
class TrainConfig:
    """Configuration for training."""
    epochs: int = 50
    lr: float = 0.001
    batch_size: int = 256
    seed: int = 42


def train_erm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    config: TrainConfig,
) -> MLP:
    """Train model with standard ERM."""
    set_seed(config.seed)

    model = MLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    n = len(x_train)
    for _ in range(config.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb, yb = x_t[idx], y_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

    return model
