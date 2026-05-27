"""Run at least three DQN experiments with different learning rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DQN.train import build_arg_parser, train_one_run


def main() -> None:
    base_parser = build_arg_parser()
    base_parser.description = __doc__
    base_parser.set_defaults(output_dir="DQN/runs_lr_sweep")
    base_parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3, 1e-4],
        help="Learning rates to sweep. The default already contains three runs.",
    )
    args = base_parser.parse_args()

    results = []
    for lr in args.learning_rates:
        run_args = argparse.Namespace(**vars(args))
        run_args.lr = lr
        if args.run_name:
            run_args.run_name = f"{args.run_name}_lr{lr:g}"
        else:
            run_args.run_name = f"{args.variant}_{args.reward_type}_lr{lr:g}_seed{args.seed}"
        print(f"\n===== Running experiment: lr={lr:g} =====")
        results.append(train_one_run(run_args))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({"sweep_summary": str(summary_path), "runs": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
