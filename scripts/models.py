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


class LogReg(nn.Module):
    """Logistic regression for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self._input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def clone(self) -> Self:
        new_model = LogReg(self._input_dim)
        new_model.load_state_dict({k: v.clone() for k, v in self.state_dict().items()})
        return new_model


class MLPMulticlass(nn.Module):
    """MLP for multiclass classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self._input_dim = input_dim
        self._num_classes = num_classes
        self._hidden_dims = hidden_dims

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x).argmax(dim=-1)

    def clone(self) -> Self:
        new_model = MLPMulticlass(self._input_dim, self._num_classes, self._hidden_dims)
        new_model.load_state_dict({k: v.clone() for k, v in self.state_dict().items()})
        return new_model


class LogRegMulticlass(nn.Module):
    """Logistic regression for multiclass classification."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x).argmax(dim=-1)

    def clone(self) -> Self:
        new_model = LogRegMulticlass(self._input_dim, self._num_classes)
        new_model.load_state_dict({k: v.clone() for k, v in self.state_dict().items()})
        return new_model


BinaryModel = MLP | LogReg
MulticlassModel = MLPMulticlass | LogRegMulticlass
AnyModel = BinaryModel | MulticlassModel


def create_model(model_type: str, input_dim: int) -> MLP | LogReg:
    """Create binary classification model by type."""
    if model_type == "mlp":
        return MLP(input_dim)
    elif model_type == "logreg":
        return LogReg(input_dim)
    raise ValueError(f"Unknown model type: {model_type}")


def create_model_multiclass(model_type: str, input_dim: int, num_classes: int) -> MLPMulticlass | LogRegMulticlass:
    """Create multiclass classification model by type."""
    if model_type == "mlp":
        return MLPMulticlass(input_dim, num_classes)
    elif model_type == "logreg":
        return LogRegMulticlass(input_dim, num_classes)
    raise ValueError(f"Unknown model type: {model_type}")


def interpolate_models(model_a: AnyModel, model_b: AnyModel, alpha: float) -> AnyModel:
    """Create interpolated model: alpha * model_a + (1 - alpha) * model_b."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    interp_state = {k: alpha * state_a[k] + (1 - alpha) * state_b[k] for k in state_a}
    new_model = model_a.clone()
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
    model_type: str = "mlp",
) -> MLP | LogReg:
    """Train binary classification model with standard ERM."""
    set_seed(config.seed)

    model = create_model(model_type, input_dim)
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


def train_erm_multiclass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    num_classes: int,
    config: TrainConfig,
    model_type: str = "mlp",
) -> MLPMulticlass | LogRegMulticlass:
    """Train multiclass classification model with standard ERM."""
    set_seed(config.seed)

    model = create_model_multiclass(model_type, input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    n = len(x_train)
    for _ in range(config.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            xb, yb = x_t[idx], y_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

    return model
