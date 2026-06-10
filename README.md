# GomokuMind - 五子棋策略评估与辅助系统

基于多策略对比的五子棋 AI 系统，通过强化学习与传统算法的对抗评估，为人类玩家提供最优落子建议。

## 项目简介

本项目实现了一个完整的五子棋（15×15）策略评估框架，包含 4 种不同策略的对比分析：

| 策略 | 类型 | 语言 | 引擎 |
|------|------|------|------|
| **启发式规则 (Heuristic)** | 传统棋型评分 | Go | Go 原生 |
| **蒙特卡洛树搜索 (MCTS)** | 树搜索 + 启发式引导 | Go | Go 原生 |
| **Q-Learning** | 经典强化学习 | Python | Go 子进程桥接 |
| **PPO (Actor-Critic)** | 现代强化学习 | Python | Go 子进程桥接 |

> Python 策略通过 `python_bridge.py` 子进程桥接接入 Go server，stdin/stdout JSON 协议通信。

## 项目结构

```
GomokuMind/
├── game-engine/           # 游戏引擎核心
│   ├── engine.go          # 15×15棋盘、落子规则、胜负判定
│   └── strategy.go        # 策略接口定义
├── strategies/            # 策略实现
│   ├── heuristic/         # 策略1: 启发式规则 (Go)
│   ├── mcts/              # 策略2: 蒙特卡洛树搜索 (Go)
│   ├── q-learning/        # 策略3: Q-Learning (Python)
│   ├── ppo/               # 策略4: PPO Actor-Critic (Python)
│   ├── python_bridge.py   # Python 策略桥接层 (Go ↔ Python)
│   ├── requirements.txt   # Python 依赖
│   └── train.py           # 训练脚本入口
├── evaluation/            # 对战评估模块 (CLI)
│   └── main.go            # 策略间对战、胜率统计
├── server/                # HTTP API 服务
│   └── main.go            # RESTful 接口，端口 8080
├── frontend/              # 前端可视化界面 (React + TypeScript + Vite)
│   ├── public/            # 静态资源 (favicon.ico)
│   └── src/
│       ├── components/    # UI 组件
│       │   ├── Board.tsx          # Canvas 绘制的 15×15 棋盘
│       │   ├── GameControl.tsx    # 游戏创建面板（模式/AI选择）
│       │   └── ThemeSwitcher.tsx   # 主题切换组件
│       ├── services/      # API 请求封装
│       │   └── api.ts
│       ├── themes/        # 主题系统
│       │   └── index.ts           # 4 主题 CSS 变量 + 棋盘配色定义
│       ├── types/         # TypeScript 类型定义
│       ├── App.tsx        # 应用入口 + 状态管理
│       └── index.tsx      # React DOM 挂载
├── go.mod                 # Go 模块配置
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
pip install -r strategies/requirements.txt

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

### 运行对战评估 (CLI)

```bash
go run ./evaluation
```

### 启动后端服务

```bash
go run ./server
```

服务启动在 `http://127.0.0.1:8080`，提供以下 REST API：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/gomoku` | 创建新游戏 `{"black_ai":"...","white_ai":"..."}` |
| GET | `/api/gomoku` | 列出所有游戏 |
| GET | `/api/gomoku/:id` | 获取游戏状态 |
| PUT | `/api/gomoku/:id` | 玩家落子 `{"row":7,"col":7}` |
| POST | `/api/gomoku/:id/ai-move` | AI 自动落子 |
| DELETE | `/api/gomoku/:id` | 删除游戏 |

### 启动前端

```bash
cd frontend
npm run dev
```

前端（`http://localhost:3000`）通过 Vite 代理将 API 请求转发到后端 8080 端口。

完整启动：

```bash
# 终端1: 启动后端
go run ./server

# 终端2: 启动前端
cd frontend && npm run dev
```

### 前端游戏模式

| 模式 | 说明 |
|------|------|
| 双人对战 (PVP) | 两个人类玩家自由博弈 |
| 人机对战 (执黑) | 人类执黑先手，AI 执白 |
| 人机对战 (执白) | 人类执白后手，AI 执黑 |
| AI 对战 | 两个 AI 自动对弈，可调节速度 |

点击「托管」按钮可让 AI 接管你的回合，再次点击取消托管。

## 策略说明

### 1. 启发式规则 (Heuristic)

基于人工设计的棋型评分表，识别活四、冲四、活三、眠三等棋型并打分。

**优势**：快速、可解释、无需训练
**劣势**：依赖人工经验，上限有限

### 2. MCTS (蒙特卡洛树搜索)

预展开候选走法，每轮模拟用 UCT 公式选择子节点，以启发式评分引导模拟结果。候选仅限已有棋子周围 2 格，大幅缩减搜索空间。

**优势**：无需训练数据，响应快（5000 次模拟 < 3ms）
**劣势**：单层搜索，深度有限

### 3. Q-Learning

经典强化学习算法，通过 Q 表学习状态-动作值函数。Go server 通过子进程调用 Python 桥接层通信。

**优势**：算法简单，RL 入门经典
**劣势**：状态空间大时收敛困难，需先训练

### 4. PPO (Actor-Critic)

现代强化学习算法，使用卷积神经网络提取棋盘特征。Go server 通过子进程调用 Python 桥接层通信。

**优势**：主流架构，适应性强
**劣势**：需要 GPU 训练，训练时间长

## 技术栈

| 模块 | 语言 | 框架/库 |
|------|------|---------|
| 游戏引擎 | Go | 标准库 |
| 启发式策略 | Go | 标准库 |
| MCTS | Go | 标准库 |
| Q-Learning | Python | NumPy |
| PPO | Python | PyTorch |
| Python 桥接 | Python | 标准库 (stdin/stdout JSON) |
| 前端界面 | TypeScript | React + Vite |

## 许可证

MIT License
