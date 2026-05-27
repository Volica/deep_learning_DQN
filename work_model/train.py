"""Train a DQN-family model on the crowdsourcing recommendation environment."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - user environment guard
    raise SystemExit(
        "PyTorch is required to train DQN. Install it first, for example: "
        "pip install torch numpy"
    ) from exc

from env import CrowdRecEnv, load_split
from DQN.agent import DQNAgent, make_agent_config
from DQN.metrics import MetricLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit(data: list[dict[str, Any]], limit: Optional[int]) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return data
    return data[:limit]


def evaluate_policy(
    agent: DQNAgent,
    data: list[dict[str, Any]],
    reward_type: str,
    max_steps: Optional[int] = None,
) -> dict[str, float]:
    env = CrowdRecEnv(
        data,
        reward_type=reward_type,
        max_candidates=agent.config.state_shape[0],
    )
    state = env.reset(shuffle=False)
    done = False
    total_reward = 0.0
    hits = 0
    invalid = 0
    steps = 0

    while not done:
        action = agent.act(state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        hits += int(info["hit"])
        invalid += int(not info["valid_action"])
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
        if next_state is not None:
            state = next_state

    return {
        "eval_steps": float(steps),
        "eval_avg_reward": total_reward / steps if steps else 0.0,
        "eval_hit_rate": hits / steps if steps else 0.0,
        "eval_invalid_action_rate": invalid / steps if steps else 0.0,
    }


def train_one_run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)

    train_data = maybe_limit(load_split(args.train_split), args.train_limit)
    val_data = maybe_limit(load_split(args.val_split), args.val_limit)
    max_candidates = max(
        max(len(sample["candidate_projects"]) for sample in train_data),
        max(len(sample["candidate_projects"]) for sample in val_data),
    )

    train_env = CrowdRecEnv(
        train_data,
        reward_type=args.reward_type,
        max_candidates=max_candidates,
        seed=args.seed,
    )
    first_state = train_env.reset(shuffle=False)
    state_shape = tuple(first_state["features"].shape)
    action_dim = int(train_env.spec.action_dim)

    agent_config = make_agent_config(
        state_shape=state_shape,
        action_dim=action_dim,
        variant=args.variant,
        lr=args.lr,
        hidden_dims=tuple(args.hidden_dims),
        network_arch=args.network_arch,
        aux_ce_weight=args.aux_ce_weight,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_replay_size=args.min_replay_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        seed=args.seed,
        device=args.device,
    )
    agent = DQNAgent(agent_config)

    run_name = args.run_name or (
        f"{args.variant}_{args.reward_type}_lr{args.lr:g}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(run_dir)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    best_val_reward = -float("inf")
    best_checkpoint = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        state = train_env.reset(shuffle=True)
        done = False
        total_reward = 0.0
        hits = 0
        invalid = 0
        losses: list[float] = []
        steps = 0

        while not done:
            action = agent.act(state, evaluate=False)
            next_state, reward, done, info = train_env.step(action)
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
                positive_action=int(info["positive_index"]),
            )
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            total_reward += reward
            hits += int(info["hit"])
            invalid += int(not info["valid_action"])
            steps += 1
            global_step += 1

            if args.max_steps_per_epoch is not None and steps >= args.max_steps_per_epoch:
                break
            if next_state is not None:
                state = next_state

        eval_stats = evaluate_policy(
            agent,
            val_data,
            reward_type=args.reward_type,
            max_steps=args.max_eval_steps,
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_steps": agent.train_steps,
            "replay_size": len(agent.replay),
            "epsilon": agent.epsilon,
            "train_avg_reward": total_reward / steps if steps else 0.0,
            "train_hit_rate": hits / steps if steps else 0.0,
            "train_invalid_action_rate": invalid / steps if steps else 0.0,
            "loss": float(np.mean(losses)) if losses else 0.0,
            **eval_stats,
        }
        logger.record(row)
        logger.write_csv()
        print(json.dumps(row, ensure_ascii=False))

        if row["eval_avg_reward"] > best_val_reward:
            best_val_reward = row["eval_avg_reward"]
            best_checkpoint = run_dir / "best_model.pt"
            agent.save(best_checkpoint, extra={"epoch": epoch, "metrics": row})

    last_checkpoint = run_dir / "last_model.pt"
    agent.save(last_checkpoint, extra={"epoch": args.epochs})
    summary = {
        "run_dir": str(run_dir),
        "best_val_reward": best_val_reward,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "last_checkpoint": str(last_checkpoint),
        "state_shape": state_shape,
        "action_dim": action_dim,
    }
    logger.write_json(summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["dqn", "double_dqn", "dueling_dqn"], default="double_dqn")
    parser.add_argument("--reward-type", choices=["worker", "requester", "hybrid"], default="worker")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--min-replay-size", type=int, default=1_000)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument(
        "--network-arch",
        choices=["candidate_scorer", "flat"],
        default="candidate_scorer",
        help="candidate_scorer shares one MLP across candidate rows; flat keeps the old flattened network.",
    )
    parser.add_argument(
        "--aux-ce-weight",
        type=float,
        default=1.0,
        help="Weight for the auxiliary positive-index cross-entropy loss. Use 0 for pure DQN.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="DQN/runs")
    parser.add_argument("--run-name", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = train_one_run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
