"""Replay buffer for off-policy DQN training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Optional

import numpy as np


@dataclass(frozen=True)
class Transition:
    state_features: np.ndarray
    state_mask: np.ndarray
    action: int
    positive_action: int
    reward: float
    next_features: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state_features: np.ndarray,
        state_mask: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
        positive_action: Optional[int] = None,
    ) -> None:
        self.buffer.append(
            Transition(
                state_features=np.asarray(state_features, dtype=np.float32).copy(),
                state_mask=np.asarray(state_mask, dtype=np.float32).copy(),
                action=int(action),
                positive_action=int(action if positive_action is None else positive_action),
                reward=float(reward),
                next_features=np.asarray(next_features, dtype=np.float32).copy(),
                next_mask=np.asarray(next_mask, dtype=np.float32).copy(),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size > len(self.buffer):
            raise ValueError("batch_size is larger than the current replay buffer size.")
        batch = self.rng.sample(list(self.buffer), int(batch_size))
        return {
            "state_features": np.stack([t.state_features for t in batch]),
            "state_mask": np.stack([t.state_mask for t in batch]),
            "actions": np.asarray([t.action for t in batch], dtype=np.int64),
            "positive_actions": np.asarray(
                [t.positive_action for t in batch], dtype=np.int64
            ),
            "rewards": np.asarray([t.reward for t in batch], dtype=np.float32),
            "next_features": np.stack([t.next_features for t in batch]),
            "next_mask": np.stack([t.next_mask for t in batch]),
            "dones": np.asarray([t.done for t in batch], dtype=np.float32),
        }
