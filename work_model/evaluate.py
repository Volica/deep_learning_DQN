"""Evaluate a trained DQN checkpoint on val or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to evaluate DQN checkpoints. Install it first: pip install torch numpy"
    ) from exc

from env import load_split
from work_model.agent import DQNAgent
from work_model.train import evaluate_policy, maybe_limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--reward-type", choices=["worker", "requester", "hybrid"], default="worker")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    agent = DQNAgent.load(args.checkpoint, device=args.device)
    data = maybe_limit(load_split(args.split), args.limit)
    metrics = evaluate_policy(agent, data, reward_type=args.reward_type, max_steps=args.max_steps)
    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "reward_type": args.reward_type,
        **metrics,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

