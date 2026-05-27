# DQN 实验方案与反思

## 任务对应关系

- 已阅读项目中的 markdown 和 Python 文件，确认现有 `env/CrowdRecEnv` 已经把众包任务推荐封装成强化学习环境。
- 使用 DQN 系列模型解决问题，代码支持 `dqn`、`double_dqn`、`dueling_dqn`。
- 算法封装在 `agent.py`、`models.py`、`replay_buffer.py` 中。
- 训练监控在 `metrics.py` 和 `train.py` 中完成，每个 run 会生成 `metrics.csv`、`config.json`、`summary.json` 和模型 checkpoint。
- `run_experiments.py` 默认运行三组学习率：`0.01`、`0.001`、`0.0001`。

## 推荐运行方式

先快速跑通：

```bash
python DQN/run_experiments.py --variant double_dqn --reward-type worker --epochs 1 --train-limit 2000 --val-limit 500 --max-steps-per-epoch 1000 --max-eval-steps 500
```

再跑完整实验：

```bash
python DQN/run_experiments.py --variant double_dqn --reward-type worker --epochs 5
```

测试集评估：

```bash
python DQN/evaluate.py --checkpoint DQN/runs_lr_sweep/double_dqn_worker_lr0.001_seed42/best_model.pt --split test --reward-type worker
```

## 效果反思

该方案能把众包推荐任务转换为“从候选 project 中选择一个 action”的 DQN 问题，状态、动作、奖励都与已有环境保持一致，因此实现成本低、复现实验方便。监控指标可以直接比较不同学习率下的 `eval_avg_reward` 和 `eval_hit_rate`，满足调参要求。

局限是当前环境基于历史日志数据构造，动作不会真正改变后续候选集分布，因此更接近离线推荐实验，而不是完全在线交互式强化学习。报告中应说明这一点，并把结论表述为“在历史候选集上的命中奖励提升”。
