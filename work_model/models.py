"""Q-network definitions used by DQN agents."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class FlatQNetwork(nn.Module):
    """Legacy DQN network that flattens all candidates before scoring actions.

    Input shape can be [batch, max_candidates, feature_dim] or already flattened.
    The output has one Q value for each candidate action slot.
    """

    def __init__(
        self,
        state_shape: tuple[int, int],
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.state_shape = tuple(state_shape)
        self.action_dim = int(action_dim)
        input_dim = int(state_shape[0] * state_shape[1])
        self.net = _build_mlp(input_dim, hidden_dims, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        return self.net(x.float())


class QNetwork(nn.Module):
    """Candidate-wise Q network.

    Each candidate row is scored by the same MLP. This matches recommendation
    data better than learning separate parameters for action slot 0, slot 1,
    ..., because the meaning of an action is the candidate's features rather
    than its padded position in the list.
    """

    def __init__(
        self,
        state_shape: tuple[int, int],
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.state_shape = tuple(state_shape)
        self.action_dim = int(action_dim)
        feature_dim = int(state_shape[1])
        self.scorer = _build_mlp(feature_dim, hidden_dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(
                "QNetwork expects input shaped [batch, max_candidates, feature_dim]."
            )
        scores = self.scorer(x.float()).squeeze(-1)
        if scores.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} action scores, got {scores.shape[1]}."
            )
        return scores


class DuelingQNetwork(nn.Module):
    """Dueling DQN network with separate value and advantage streams."""

    def __init__(
        self,
        state_shape: tuple[int, int],
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.state_shape = tuple(state_shape)
        self.action_dim = int(action_dim)
        input_dim = int(state_shape[0] * state_shape[1])
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer for DuelingQNetwork.")

        trunk_layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            trunk_layers.append(nn.Linear(last_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            last_dim = hidden_dim

        shared_dim = hidden_dims[-1]
        trunk_layers.append(nn.Linear(last_dim, shared_dim))
        trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.value = nn.Linear(shared_dim, 1)
        self.advantage = nn.Linear(shared_dim, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        z = self.trunk(x.float())
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
