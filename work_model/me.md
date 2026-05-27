5/25 晚上跑的结果
             
(rl_env) PS E:\系统默认\桌面\Sec\研一下\强化学习\强化学习\processed_data> python DQN/train.py --variant double_dqn --reward-type worker --lr 0.001 --epochs 5
{"epoch": 1, "global_step": 134475, "train_steps": 133476, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.24571109871723368, "train_hit_rate": 0.24571109871723368, "train_invalid_action_rate": 0.0, "loss": 2.5824591785686297, "eval_steps": 28816.0, "eval_avg_reward": 0.19218489727928928, "eval_hit_rate": 0.19218489727928928, "eval_invalid_action_rate": 0.0}
{"epoch": 2, "global_step": 268950, "train_steps": 267951, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.286395240751069, "train_hit_rate": 0.286395240751069, "train_invalid_action_rate": 0.0, "loss": 2.424652279843892, "eval_steps": 28816.0, "eval_avg_reward": 0.18229455857856747, "eval_hit_rate": 0.18229455857856747, "eval_invalid_action_rate": 0.0}
{"epoch": 3, "global_step": 403425, "train_steps": 402426, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.29702918758133484, "train_hit_rate": 0.29702918758133484, "train_invalid_action_rate": 0.0, "loss": 2.3924288706984256, "eval_steps": 28816.0, "eval_avg_reward": 0.16681704608550804, "eval_hit_rate": 0.16681704608550804, "eval_invalid_action_rate": 0.0}
{"epoch": 4, "global_step": 537900, "train_steps": 536901, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.30402677077523704, "train_hit_rate": 0.30402677077523704, "train_invalid_action_rate": 0.0, "loss": 2.369185114028308, "eval_steps": 28816.0, "eval_avg_reward": 0.16383259300388672, "eval_hit_rate": 0.16383259300388672, "eval_invalid_action_rate": 0.0}
{"epoch": 5, "global_step": 672375, "train_steps": 671376, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.3080944413459751, "train_hit_rate": 0.3080944413459751, "train_invalid_action_rate": 0.0, "loss": 2.3532382933484906, "eval_steps": 28816.0, "eval_avg_reward": 0.15873126041088284, "eval_hit_rate": 0.15873126041088284, "eval_invalid_action_rate": 0.0}
{
  "run_dir": "DQN\\runs\\double_dqn_worker_lr0.001_20260525_205257",
  "best_val_reward": 0.19218489727928928,
  "best_checkpoint": "DQN\\runs\\double_dqn_worker_lr0.001_20260525_205257\\best_model.pt",
  "last_checkpoint": "DQN\\runs\\double_dqn_worker_lr0.001_20260525_205257\\last_model.pt",
  "state_shape": [
    21,
    11
  ],
  "action_dim": 21
}

5/26白天跑的结果

(rl_env) PS E:\系统默认\桌面\Sec\研一下\强化学习\强化学习\processed_data> python DQN/run_experiments.py --variant double_dqn --reward-type worker --epochs 5

===== Running experiment: lr=0.01 =====
{"epoch": 1, "global_step": 134475, "train_steps": 133476, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.23031790295593976, "train_hit_rate": 0.23031790295593976, "train_invalid_action_rate": 0.0, "loss": 2.651936405046717, "eval_steps": 28816.0, "eval_avg_reward": 0.21817740144364242, "eval_hit_rate": 0.21817740144364242, "eval_invalid_action_rate": 0.0}
{"epoch": 2, "global_step": 268950, "train_steps": 267951, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.24864101134039784, "train_hit_rate": 0.24864101134039784, "train_invalid_action_rate": 0.0, "loss": 2.567808340205249, "eval_steps": 28816.0, "eval_avg_reward": 0.221682398667407, "eval_hit_rate": 0.221682398667407, "eval_invalid_action_rate": 0.0}
{"epoch": 3, "global_step": 403425, "train_steps": 402426, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.24870793827849041, "train_hit_rate": 0.24870793827849041, "train_invalid_action_rate": 0.0, "loss": 2.5667783573166236, "eval_steps": 28816.0, "eval_avg_reward": 0.21810799555802332, "eval_hit_rate": 0.21810799555802332, "eval_invalid_action_rate": 0.0}
{"epoch": 4, "global_step": 537900, "train_steps": 536901, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.2483733035880275, "train_hit_rate": 0.2483733035880275, "train_invalid_action_rate": 0.0, "loss": 2.565738710544215, "eval_steps": 28816.0, "eval_avg_reward": 0.22050249861188229, "eval_hit_rate": 0.22050249861188229, "eval_invalid_action_rate": 0.0}
{"epoch": 5, "global_step": 672375, "train_steps": 671376, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.2471611823759063, "train_hit_rate": 0.2471611823759063, "train_invalid_action_rate": 0.0, "loss": 2.569132416523573, "eval_steps": 28816.0, "eval_avg_reward": 0.22081482509716824, "eval_hit_rate": 0.22081482509716824, "eval_invalid_action_rate": 0.0}

===== Running experiment: lr=0.001 =====
{"epoch": 1, "global_step": 134475, "train_steps": 133476, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.24571109871723368, "train_hit_rate": 0.24571109871723368, "train_invalid_action_rate": 0.0, "loss": 2.5824591785686297, "eval_steps": 28816.0, "eval_avg_reward": 0.19218489727928928, "eval_hit_rate": 0.19218489727928928, "eval_invalid_action_rate": 0.0}
{"epoch": 2, "global_step": 268950, "train_steps": 267951, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.286395240751069, "train_hit_rate": 0.286395240751069, "train_invalid_action_rate": 0.0, "loss": 2.424652279843892, "eval_steps": 28816.0, "eval_avg_reward": 0.18229455857856747, "eval_hit_rate": 0.18229455857856747, "eval_invalid_action_rate": 0.0}
{"epoch": 3, "global_step": 403425, "train_steps": 402426, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.29702918758133484, "train_hit_rate": 0.29702918758133484, "train_invalid_action_rate": 0.0, "loss": 2.3924288706984256, "eval_steps": 28816.0, "eval_avg_reward": 0.16681704608550804, "eval_hit_rate": 0.16681704608550804, "eval_invalid_action_rate": 0.0}
{"epoch": 4, "global_step": 537900, "train_steps": 536901, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.30402677077523704, "train_hit_rate": 0.30402677077523704, "train_invalid_action_rate": 0.0, "loss": 2.369185114028308, "eval_steps": 28816.0, "eval_avg_reward": 0.16383259300388672, "eval_hit_rate": 0.16383259300388672, "eval_invalid_action_rate": 0.0}
{"epoch": 5, "global_step": 672375, "train_steps": 671376, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.3080944413459751, "train_hit_rate": 0.3080944413459751, "train_invalid_action_rate": 0.0, "loss": 2.3532382933484906, "eval_steps": 28816.0, "eval_avg_reward": 0.15873126041088284, "eval_hit_rate": 0.15873126041088284, "eval_invalid_action_rate": 0.0}

===== Running experiment: lr=0.0001 =====
{"epoch": 1, "global_step": 134475, "train_steps": 133476, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.23493586168432795, "train_hit_rate": 0.23493586168432795, "train_invalid_action_rate": 0.0, "loss": 2.654616108814211, "eval_steps": 28816.0, "eval_avg_reward": 0.20873820099944476, "eval_hit_rate": 0.20873820099944476, "eval_invalid_action_rate": 0.0}
{"epoch": 2, "global_step": 268950, "train_steps": 267951, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.2595872838817624, "train_hit_rate": 0.2595872838817624, "train_invalid_action_rate": 0.0, "loss": 2.5393318655322887, "eval_steps": 28816.0, "eval_avg_reward": 0.1971821210438645, "eval_hit_rate": 0.1971821210438645, "eval_invalid_action_rate": 0.0}
{"epoch": 3, "global_step": 403425, "train_steps": 402426, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.2673879903327756, "train_hit_rate": 0.2673879903327756, "train_invalid_action_rate": 0.0, "loss": 2.5129538635446096, "eval_steps": 28816.0, "eval_avg_reward": 0.1962104386451971, "eval_hit_rate": 0.1962104386451971, "eval_invalid_action_rate": 0.0}
{"epoch": 4, "global_step": 537900, "train_steps": 536901, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.2739542665923034, "train_hit_rate": 0.2739542665923034, "train_invalid_action_rate": 0.0, "loss": 2.4894105215998117, "eval_steps": 28816.0, "eval_avg_reward": 0.1854872293170461, "eval_hit_rate": 0.1854872293170461, "eval_invalid_action_rate": 0.0}
{"epoch": 5, "global_step": 672375, "train_steps": 671376, "replay_size": 50000, "epsilon": 0.050000000000000044, "train_avg_reward": 0.27913738613125116, "train_hit_rate": 0.27913738613125116, "train_invalid_action_rate": 0.0, "loss": 2.4732159538467613, "eval_steps": 28816.0, "eval_avg_reward": 0.18507079400333148, "eval_hit_rate": 0.18507079400333148, "eval_invalid_action_rate": 0.0}
{
  "sweep_summary": "DQN\\runs_lr_sweep\\sweep_summary.json",
  "runs": [
    {
      "run_dir": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.01_seed42",
      "best_val_reward": 0.221682398667407,
      "best_checkpoint": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.01_seed42\\best_model.pt",
      "last_checkpoint": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.01_seed42\\last_model.pt",
      "state_shape": [
        21,
        11
      ],
      "action_dim": 21
    },
    {
      "run_dir": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.001_seed42",
      "best_val_reward": 0.19218489727928928,
      "best_checkpoint": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.001_seed42\\best_model.pt",
      "last_checkpoint": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.001_seed42\\last_model.pt",
      "state_shape": [
        21,
        11
      ],
      "action_dim": 21
    },
    {
      "run_dir": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.0001_seed42",
      "best_val_reward": 0.20873820099944476,
      "best_checkpoint": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.0001_seed42\\best_model.pt",
      "last_checkpoint": "DQN\\runs_lr_sweep\\double_dqn_worker_lr0.0001_seed42\\last_model.pt",
      "state_shape": [
        21,
        11
      ],
      "action_dim": 21
    }
  ]
}