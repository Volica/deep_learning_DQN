# 众包任务推荐 DQN 强化学习项目

本项目尝试将 DQN 系列强化学习方法用于众包平台中的任务推荐问题。项目从两个视角建模推荐任务：

- **Worker 视角**：当一个 worker 到达平台时，从候选 project 中推荐一个最可能被其选择的任务。
- **Requester 视角**：当一个 project 需要更多参与者时，从候选 worker 中推荐一个更合适的 worker。

项目包含数据预处理、强化学习环境封装、DQN/Double DQN/Dueling DQN 模型训练、评估和可视化分析代码。

## 项目背景

众包平台的推荐目标不是单一的点击率问题。一方面，worker 希望看到自己更感兴趣、更可能完成的任务；另一方面，requester 希望任务能得到高质量、多样化、及时的回答。尤其当任务临近截止时间时，系统更需要优先推荐合适的高质量 worker。

因此，本项目将历史交互日志包装成离线强化学习环境。每条历史事件视为一个决策点，环境返回当前候选集特征、合法动作掩码、奖励和下一条事件状态，模型学习在候选列表中选择一个推荐对象。

## 项目结构

```text
.
├── data/                         # 原始数据
│   ├── project/                  # project 原始信息
│   ├── entry/                    # worker 参与记录
│   ├── project_list.csv          # project 列表
│   └── worker_quality.csv        # worker 质量分数
├── data_processed/               # 数据预处理脚本和 pkl 数据
│   ├── data_process.py           # worker 视角数据处理
│   ├── data_process_request.py   # requester 视角数据处理
│   ├── enhanced_train.pkl        # worker 训练集
│   ├── enhanced_val.pkl          # worker 验证集
│   ├── enhanced_test.pkl         # worker 测试集
│   ├── requester_train.pkl       # requester 训练集
│   ├── requester_val.pkl         # requester 验证集
│   ├── requester_test.pkl        # requester 测试集
│   ├── worker_episodes.pkl       # 按 worker_id 组织的序列数据
│   ├── project_episodes.pkl      # 按 project_id 组织的序列数据
│   ├── worker_features.pkl       # worker 静态特征
│   ├── norm.json                 # 归一化参数
│   └── category_maps.json        # 类别、子类别、行业映射
├── env/                          # 强化学习环境
│   ├── crowd_env.py              # CrowdRecEnv 和 RequesterCrowdEnv
│   └── demo_random.py            # 随机策略测试环境是否可运行
├── work_model/                   # DQN 主体模型代码
│   ├── models.py                 # Q 网络与 Dueling Q 网络
│   ├── agent.py                  # DQN Agent
│   ├── replay_buffer.py          # 经验回放池
│   ├── train.py                  # 单次训练入口
│   ├── evaluate.py               # 模型评估入口
│   └── run_experiments.py        # 批量实验入口
├── requester_model/              # worker/requester 对比实验与可视化
│   ├── visualize_train.py        # 主训练与可视化脚本
│   ├── run_all_variants.py       # 批量运行 DQN 变体
│   └── plot_from_csv.py          # 从 metrics.csv 重新绘图
├── 报告.md                       # 项目报告
└── 强化学习大作业.docx            # 课程文档
```

## 数据说明

预处理后的数据统一放在 `data_processed/` 目录下。

Worker 视角数据：

```text
enhanced_train.pkl
enhanced_val.pkl
enhanced_test.pkl
worker_episodes.pkl
worker_features.pkl
```

Requester 视角数据：

```text
requester_train.pkl
requester_val.pkl
requester_test.pkl
project_episodes.pkl
worker_features.pkl
```

Worker 视角中，每条样本表示一个 worker 到达平台，系统从候选 project 中推荐一个任务。Requester 视角中，每条样本表示一个 project 在当前时刻需要推荐 worker，系统从候选 worker 中选择一个推荐对象。

## 强化学习环境

环境核心代码位于 `env/crowd_env.py`。

| 环境 | 视角 | 候选对象 | 状态形状 | 数据文件 |
| ---- | ---- | ---- | ---- | ---- |
| `CrowdRecEnv` | Worker | project | `[20, 13]` | `enhanced_train/val/test.pkl` |
| `RequesterCrowdEnv` | Requester | worker | `[20, 16]` | `requester_train/val/test.pkl` |

两个环境都使用 `action_mask` 屏蔽 padding 位置和非法候选项。动作统一表示候选列表下标：

```text
action ∈ [0, 19]
```

## 模型方法

项目支持三种 DQN 系列模型：

| 模型 | 说明 |
| ---- | ---- |
| `dqn` | 基础 DQN |
| `double_dqn` | Double DQN，分离动作选择和目标估值 |
| `dueling_dqn` | Dueling Double DQN，拆分状态价值和动作优势 |

模型采用候选对象共享打分结构：

```text
Q(s, a_i) = MLP(feature_i)
```

也就是说，网络对每个候选 project 或 candidate worker 的特征向量分别打分，最后选择 Q 值最高且合法的候选对象。

## 安装依赖

建议使用 Python 3.10 或以上版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch pandas matplotlib tqdm python-dateutil
```

如果只运行 `work_model/` 中的基础训练代码，也可以安装最小依赖：

```bash
pip install -r work_model/requirements.txt
```

## 运行方式

训练 worker 目标：

```bash
python3 work_model/train.py \
  --variant double_dqn \
  --reward-type worker \
  --lr 0.001 \
```

评估已保存模型：

```bash
python3 work_model/evaluate.py \
  --checkpoint work_model/runs/<run_name>/best_model.pt \
  --split test \
  --reward-type worker
```

## Worker 与 Requester 对比实验

`requester_model/visualize_train.py` 可以训练 worker 和 requester 两个目标，并生成训练曲线和测试结果图。

只训练 worker 目标：

```bash
python3 requester_model/visualize_train.py \
  --variant double_dqn \
  --lr 0.001 \
  --skip-requester
```

只训练 requester 目标：

```bash
python3 requester_model/visualize_train.py \
  --variant double_dqn \
  --lr 0.001 \
  --skip-worker
```

同时训练 worker 和 requester 目标：

```bash
python3 requester_model/visualize_train.py \
  --variant double_dqn \
  --lr 0.001
```

批量运行 DQN、Double DQN、Dueling DQN：

```bash
python3 requester_model/run_all_variants.py --lr 0.001
```

## 输出文件

训练过程会在对应的 runs 目录下保存：

```text
config.json
metrics.csv
summary.json
best_model.pt
last_model.pt
training_curve_*.png
test_result.json
```

常用指标包括：

- `train_avg_reward`
- `train_hit_rate`
- `eval_avg_reward`
- `eval_hit_rate`
- `eval_invalid_action_rate`
- `loss`
- `epsilon`

Requester 目标还会额外记录推荐 worker 的平均质量、多样性比例和紧急任务下的 worker 质量。

## 注意事项

- 本项目是基于历史日志的离线强化学习实验，环境会按时间顺序推进事件，但模型动作不会真实改变原始数据。
- 全量训练数据较大，调试时可以使用 `--train-limit`、`--val-limit`、`--max-steps-per-epoch` 和 `--max-eval-steps` 限制运行规模。
- Requester 视角依赖 `requester_train.pkl`、`requester_val.pkl`、`requester_test.pkl` 和 `project_episodes.pkl`。
