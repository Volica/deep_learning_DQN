已改代码，目标是让验证命中率真正往上走。

主要改了四处：

- [DQN/models.py]：新增候选级共享打分网络 `QNetwork`，每个 candidate 用同一套 MLP 打分，比原来把 21 个候选拍平后按槽位输出更适合推荐任务。
- [DQN/agent.py]：默认 `gamma=0.0`，因为这个环境每条样本基本是独立推荐决策，不应该把下一条无关样本的 Q 值传回来。
- [DQN/agent.py]：加入辅助交叉熵损失，用训练数据里的 `positive_index` 直接优化命中率。
- [DQN/replay_buffer.py]和 [DQN/train.py]：把 `positive_index` 存进 replay buffer 并参与训练。

你原来的命令可以直接重跑，现在默认就会使用新结构：

```bash
python DQN/train.py --variant double_dqn --reward-type worker --lr 0.001 --epochs 5
```

建议再跑一组学习率搜索：

```bash
python DQN/run_experiments.py --variant double_dqn --reward-type worker --
epochs 5
```

如果想对比“旧代码逻辑”，可以用：

```bash
python DQN/train.py --variant double_dqn --reward-type worker --lr 0.001 --epochs 5 --network-arch flat --aux-ce-weight 0 --gamma 0.99
```

我这边没法实际训练验证，因为当前 Codex shell 没有可用的 `python`/`conda`，但代码层面的调用链已经检查过。注意新版本的 `loss` 包含辅助分类损失，数值不能和旧版 loss 直接比较，重点看 `eval_hit_rate`。