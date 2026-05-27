"""DQN, Double DQN, and Dueling DQN agent implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .models import DuelingQNetwork, FlatQNetwork, QNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class AgentConfig:
    state_shape: tuple[int, int]
    action_dim: int
    lr: float = 1e-3
    gamma: float = 0.0
    batch_size: int = 64
    buffer_size: int = 50_000
    min_replay_size: int = 1_000
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    hidden_dims: tuple[int, ...] = (256, 128)
    double_dqn: bool = True
    dueling: bool = False
    network_arch: str = "candidate_scorer"
    aux_ce_weight: float = 1.0
    grad_clip_norm: float = 10.0
    seed: int = 42
    device: str = "auto"


class DQNAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        if config.dueling:
            network_cls = DuelingQNetwork
        elif config.network_arch == "flat":
            network_cls = FlatQNetwork
        elif config.network_arch == "candidate_scorer":
            network_cls = QNetwork
        else:
            raise ValueError(
                "network_arch must be one of: candidate_scorer, flat."
            )
        self.online_net = network_cls(
            state_shape=config.state_shape,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.target_net = network_cls(
            state_shape=config.state_shape,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(config.buffer_size, seed=config.seed)
        self.train_steps = 0
        self.env_steps = 0
        self.epsilon = float(config.epsilon_start)

    def act(self, state: Mapping[str, np.ndarray], evaluate: bool = False) -> int:
        legal_actions = np.flatnonzero(np.asarray(state["action_mask"]) > 0)
        if len(legal_actions) == 0:
            raise RuntimeError("No legal action is available.")

        epsilon = 0.0 if evaluate else self.epsilon
        if self.rng.random() < epsilon:
            return int(self.rng.choice(legal_actions))

        with torch.no_grad():
            features = torch.as_tensor(
                state["features"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.online_net(features).squeeze(0).cpu().numpy()

        q_values[np.asarray(state["action_mask"]) <= 0] = -1e9
        return int(np.argmax(q_values))

    def remember(
        self,
        state: Mapping[str, np.ndarray],
        action: int,
        reward: float,
        next_state: Optional[Mapping[str, np.ndarray]],
        done: bool,
        positive_action: Optional[int] = None,
    ) -> None:
        if next_state is None:
            next_features = np.zeros_like(state["features"], dtype=np.float32)
            next_mask = np.zeros_like(state["action_mask"], dtype=np.float32)
        else:
            next_features = next_state["features"]
            next_mask = next_state["action_mask"]

        self.replay.push(
            state["features"],
            state["action_mask"],
            action,
            reward,
            next_features,
            next_mask,
            done,
            positive_action=positive_action,
        )
        self.env_steps += 1
        self._update_epsilon()

    def learn(self) -> Optional[float]:
        if len(self.replay) < max(self.config.batch_size, self.config.min_replay_size):
            return None

        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(batch["state_features"], dtype=torch.float32, device=self.device)
        state_masks = torch.as_tensor(batch["state_mask"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        positive_actions = torch.as_tensor(
            batch["positive_actions"], dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_features"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_mask"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_values = self.online_net(states)
        q_selected = q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                next_online_q = self.online_net(next_states)
                next_online_q = next_online_q.masked_fill(next_masks <= 0, -1e9)
                next_actions = next_online_q.argmax(dim=1, keepdim=True)
                next_target_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_target_q_all = self.target_net(next_states)
                next_target_q_all = next_target_q_all.masked_fill(next_masks <= 0, -1e9)
                next_target_q = next_target_q_all.max(dim=1).values

            target_q = rewards + self.config.gamma * (1.0 - dones) * next_target_q

        td_loss = self.loss_fn(q_selected, target_q)
        if self.config.aux_ce_weight > 0:
            masked_logits = q_values.masked_fill(state_masks <= 0, -1e9)
            ce_loss = nn.functional.cross_entropy(masked_logits, positive_actions)
            loss = td_loss + self.config.aux_ce_weight * ce_loss
        else:
            loss = td_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.sync_target_network()

        return float(loss.detach().cpu().item())

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str | Path, extra: Optional[dict[str, Any]] = None) -> None:
        payload = {
            "agent_config": asdict(self.config),
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "env_steps": self.env_steps,
            "epsilon": self.epsilon,
            "extra": extra or {},
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "DQNAgent":
        payload = torch.load(Path(path), map_location="cpu")
        cfg_dict = payload["agent_config"]
        cfg_dict["state_shape"] = tuple(cfg_dict["state_shape"])
        cfg_dict["hidden_dims"] = tuple(cfg_dict["hidden_dims"])
        if "network_arch" not in cfg_dict:
            state_dict = payload.get("online_state_dict", {})
            cfg_dict["network_arch"] = (
                "flat" if any(key.startswith("net.") for key in state_dict) else "candidate_scorer"
            )
        cfg_dict["device"] = device
        agent = cls(AgentConfig(**cfg_dict))
        agent.online_net.load_state_dict(payload["online_state_dict"])
        agent.target_net.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.train_steps = int(payload.get("train_steps", 0))
        agent.env_steps = int(payload.get("env_steps", 0))
        agent.epsilon = float(payload.get("epsilon", agent.config.epsilon_end))
        return agent

    def _update_epsilon(self) -> None:
        fraction = min(1.0, self.env_steps / max(1, self.config.epsilon_decay_steps))
        self.epsilon = self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )


def make_agent_config(
    state_shape: tuple[int, int],
    action_dim: int,
    variant: str,
    lr: float,
    hidden_dims: Sequence[int],
    network_arch: str = "candidate_scorer",
    aux_ce_weight: float = 1.0,
    **kwargs: Any,
) -> AgentConfig:
    variant = variant.lower()
    if variant not in {"dqn", "double_dqn", "dueling_dqn"}:
        raise ValueError("variant must be one of: dqn, double_dqn, dueling_dqn.")
    return AgentConfig(
        state_shape=tuple(state_shape),
        action_dim=int(action_dim),
        lr=float(lr),
        hidden_dims=tuple(hidden_dims),
        double_dqn=variant in {"double_dqn", "dueling_dqn"},
        dueling=variant == "dueling_dqn",
        network_arch=network_arch,
        aux_ce_weight=float(aux_ce_weight),
        **kwargs,
    )
