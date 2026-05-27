# DQN 众包任务推荐实验

这个目录把现有 `env/CrowdRecEnv` 封装成可训练的 DQN 系列实验代码，支持：

- `dqn`：基础 DQN
- `double_dqn`：Double DQN，默认推荐
- `dueling_dqn`：Dueling Double DQN

## 文件说明

- `models.py`：QNetwork 和 DuelingQNetwork。
- `replay_buffer.py`：经验回放池。
- `agent.py`：DQN Agent，包含 epsilon-greedy、mask 动作选择、TD 更新、target network。
- `train.py`：单次训练入口。
- `evaluate.py`：加载 checkpoint 后在 train/val/test 上评估。
- `run_experiments.py`：自动运行至少三组学习率实验。
- `metrics.py`：把训练过程写入 CSV/JSON。
- `requirements.txt`：需要 `numpy` 和 `torch`。


## 打开anaconda prompt，进入项目文件夹

```bash
E：
cd E:\系统默认\桌面\Sec\研一下\强化学习\强化学习\processed_data
```

## 创建虚拟环境
```bash
conda create -n rl_env python=3.10 -y
conda activate rl_env
```

## 安装依赖

当前代码使用 PyTorch：

```bash
pip install -r DQN/requirements.txt
pip install pandas
```

## 单次训练

```bash
python DQN/train.py --variant double_dqn --reward-type worker --lr 0.001 --epochs 5
```

为了快速检查流程，可以限制数据规模：

```bash
python DQN/train.py --variant double_dqn --reward-type worker --lr 0.001 --epochs 1 --train-limit 2000 --val-limit 500 --max-steps-per-epoch 1000 --max-eval-steps 500
```

## 三组学习率实验

默认会跑 `0.01, 0.001, 0.0001` 三组：

```bash
python DQN/run_experiments.py --variant double_dqn --reward-type worker --epochs 5
```

快速版：

```bash
python DQN/run_experiments.py --variant double_dqn --reward-type worker --epochs 1 --train-limit 2000 --val-limit 500 --max-steps-per-epoch 1000 --max-eval-steps 500
```

## 评估模型

```bash
python DQN/evaluate.py --checkpoint DQN/runs_lr_sweep/double_dqn_worker_lr0.001_seed42/best_model.pt --split test --reward-type worker
```

## 监控指标

每个 run 会输出：

- `metrics.csv`：每个 epoch 的训练和验证指标。
- `config.json`：本次运行参数。
- `summary.json`：最佳验证奖励和 checkpoint 路径。
- `best_model.pt` / `last_model.pt`：模型权重。

核心指标包括：

- `train_avg_reward`
- `train_hit_rate`
- `train_invalid_action_rate`
- `eval_avg_reward`
- `eval_hit_rate`
- `eval_invalid_action_rate`
- `loss`
- `epsilon`

## 建模说明

状态使用 `state["features"]`，即 `[max_candidates, feature_dim]` 的候选项目特征矩阵；动作是推荐候选项目下标；动作选择会使用 `state["action_mask"]` 屏蔽 padding 位置。奖励由环境参数 `reward_type` 控制，可选择参与者目标、请求者目标或混合目标。
