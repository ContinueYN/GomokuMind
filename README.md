# GomokuMind - 五子棋策略评估与辅助系统

基于多策略对比的五子棋 AI 系统，通过强化学习与传统算法的对抗评估，为人类玩家提供最优落子建议。

## 项目简介

本项目实现了一个完整的五子棋（15×15）策略评估框架，包含 4 种不同策略的对比分析：

| 策略 | 类型 | 语言 | 训练需求 |
|------|------|------|----------|
| **启发式规则** | 传统棋型评分 | Go | 无需训练 |
| **Q-Learning** | 经典强化学习 | Python | 需训练 |
| **PPO (Actor-Critic)** | 现代强化学习 | Python | 需训练 |
| **MCTS** | 蒙特卡洛树搜索 | Go | 无需训练 |

## 项目结构

```
GomokuMind/
├── game-engine/           # 游戏引擎核心
│   ├── engine.go          # 15×15棋盘、落子规则、胜负判定
│   └── strategy.go        # 策略接口定义
├── strategies/            # 策略实现
│   ├── heuristic/         # 策略1: 启发式规则 (Go)
│   ├── mcts/              # 策略4: 蒙特卡洛树搜索 (Go)
│   ├── q-learning/        # 策略2: Q-Learning (Python)
│   ├── ppo/               # 策略3: PPO Actor-Critic (Python)
│   ├── requirements.txt   # Python依赖
│   └── train.py           # 训练脚本入口
├── evaluation/            # 对战评估模块
│   └── main.go            # 策略间对战、胜率统计
├── frontend/              # 前端可视化界面
│   └── src/               # React + TypeScript + Vite
├── go.mod                 # Go模块配置
└── README.md              # 项目说明
```

## 快速开始

### 环境要求

- Go 1.21+
- Python 3.10+
- Node.js 18+ (前端)

### 安装依赖

```bash
# Go 依赖
go mod tidy

# Python 依赖
pip install -r strategies/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 前端依赖
cd frontend && npm install
```

### 训练策略

```bash
# 训练所有策略
python strategies/train.py --strategy all --episodes 1000

# 仅训练 Q-Learning
python strategies/train.py --strategy q-learning --episodes 1000

# 仅训练 PPO
python strategies/train.py --strategy ppo --episodes 500
```

### 运行对战评估

```bash
cd evaluation
go run main.go
```

### 启动前端

```bash
cd frontend
npm run dev
```

## 策略说明

### 1. 启发式规则 (Heuristic)

基于人工设计的棋型评分表，识别活四、冲四、活三、眠三等棋型并打分。

**优势**：快速、可解释、无需训练
**劣势**：依赖人工经验，上限有限

### 2. Q-Learning

经典强化学习算法，通过 Q 表学习状态-动作值函数。使用区域抽象处理 15×15 棋盘的大状态空间。

**优势**：算法简单，RL 入门经典
**劣势**：状态空间大时收敛困难

### 3. PPO (Actor-Critic)

现代强化学习算法，使用卷积神经网络提取棋盘特征，Actor 输出策略概率 + Critic 评估状态价值。

**优势**：2025-2026 主流架构，适应性强
**劣势**：需要 GPU 训练，训练时间长

### 4. MCTS (蒙特卡洛树搜索)

通过随机模拟评估落子价值，使用 UCT 算法平衡探索与利用。

**优势**：无需训练数据，结果可靠
**劣势**：搜索深度受限于计算资源

## 技术栈

| 模块 | 语言 | 框架/库 |
|------|------|---------|
| 游戏引擎 | Go | 标准库 |
| 启发式策略 | Go | 标准库 |
| MCTS | Go | 标准库 |
| Q-Learning | Python | NumPy |
| PPO | Python | PyTorch |
| 前端界面 | TypeScript | React + Vite |

## 许可证

MIT License
