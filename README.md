# GomokuMind - 五子棋策略评估与辅助系统

基于多策略对比的五子棋 AI 系统，通过强化学习与传统算法的对抗评估，为人类玩家提供最优落子建议。

## 项目简介

本项目实现了一个完整的五子棋（15×15）策略评估框架，包含 5 种不同策略的对比分析：

| 策略 | 类型 | 语言 | 引擎 |
|------|------|------|------|
| **启发式规则 (Heuristic)** | 传统棋型评分 | Go | Go 原生 |
| **蒙特卡洛树搜索 (MCTS)** | 树搜索 + 启发式引导 | Go | Go 原生 |
| **Alpha-Beta** | 博弈树搜索 | Go | Go 原生 |
| **Minimax** | Negamax + Alpha-Beta 剪枝 | Go + Python | Go 持久子进程 |
| **AlphaZero** | 深度强化学习 | Go + Python | Go 持久子进程 |

> Minimax 和 AlphaZero 策略通过 Go 端管理持久 Python 子进程管道通信；`python_bridge.py` 提供独立 CLI 调用备用入口。

## 策略锦标赛结果

83 场 AI vs AI 双循环对战（每对策略交战 4 轮，轮流执黑执白）：

| 排名 | 策略 | 胜 | 负 | 平 | 胜率 |
|------|------|----|----|----|------|
| 🥇 第1 | **AlphaZero** | 20 | 11 | 1 | **62.5%** |
| 🥈 第2 | MCTS | 19 | 9 | 6 | 55.9% |
| 🥉 第3 | Heuristic | 19 | 12 | 4 | 54.3% |
| 第4 | Minimax | 17 | 14 | 1 | 53.1% |
| 第5 | AlphaBeta | 2 | 31 | 0 | 6.1% |

> 完整对局记录见 `records.json`（90 条历史对局，含人类对战）。

**关键发现**：
- **AlphaZero** 统计胜率最高 (62.5%)，执黑 88% 胜率，但依赖 PyTorch + Python 子进程
- **MCTS** 最稳健 —— 负场最少 (9)、两面均衡（执黑 77% / 执白 60%），纯 Go 实现 3ms 响应
- 15×15 无禁手规则下**先手优势极大**，所有策略执黑胜率远高于执白

## 项目结构

```
GomokuMind/
├── game-engine/           # 游戏引擎核心
│   ├── engine.go          # 15×15棋盘、落子规则、胜负判定
│   └── strategy.go        # 策略接口定义
├── strategies/            # 策略实现
│   ├── heuristic/         # 策略1: 启发式规则 (Go)
│   ├── mcts/              # 策略2: 蒙特卡洛树搜索 (Go)
│   ├── alphabeta/         # 策略3: Alpha-Beta 搜索 (Go)
│   ├── minimax/           # 策略4: Minimax 剪枝搜索 (Go + Python)
│   │   ├── minimax.go     # Go 策略包装器
│   │   └── minimax.py     # Python 持久推理服务
│   ├── alphazero/         # 策略5: AlphaZero 深度学习 (Go + Python)
│   │   ├── alphazero.go   # Go 策略包装器
│   │   ├── infer.py       # Python 持久推理服务
│   │   ├── gomoku.py      # 游戏环境
│   │   └── alphazero_train.py  # 训练脚本
│   ├── python_bridge.py   # Python 策略桥接层 (Go ↔ Python)
│   └── requirements.txt   # Python 依赖
├── server/                # HTTP API 服务
│   ├── main.go            # RESTful 接口，端口 8080
│   └── records.go         # 对局记录持久化
├── frontend/              # 前端可视化界面 (React + TypeScript + Vite)
│   ├── public/            # 静态资源 (poem SVG, favicon)
│   └── src/
│       ├── components/    # UI 组件
│       │   ├── Board.tsx          # Canvas 绘制的 15×15 棋盘
│       │   ├── GameControl.tsx    # 游戏创建面板（模式/AI选择）
│       │   ├── PoemDisplay.tsx    # 诗句轮播背景层
│       │   └── ThemeSwitcher.tsx  # 主题切换组件
│       ├── services/      # API 请求封装
│       │   └── api.ts
│       ├── themes/        # 主题系统
│       │   └── index.ts           # 4 主题 CSS 变量 + 棋盘配色定义
│       ├── types/         # TypeScript 类型定义
│       ├── App.tsx        # 应用入口 + 状态管理
│       └── index.tsx      # React DOM 挂载
├── records.json           # 对局历史记录 (90条)
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
# 训练 AlphaZero 模型（深度强化学习）
python strategies/alphazero/alphazero_train.py                     # 从头训练
python strategies/alphazero/alphazero_train.py --resume best       # 从最优 checkpoint 恢复
python strategies/alphazero/alphazero_train.py --iters 500 --eps 200 --sims 400
```

### 启动后端服务

```bash
go run ./server
# 或编译后运行
go build -o server.exe ./server && ./server.exe
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
| GET | `/api/stats` | 对局统计（总局数、胜率、AI 排行） |
| GET | `/api/records?limit=50` | 对局历史记录 |

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

### 界面特色

- **四时主题** — 春之呼吸、秋日郊野、寒冬低语、繁星守望四种配色，一键切换
- **诗句轮播** — 棋盘两侧各有一列古风诗句沿竖向轨道循环滚动，随主题变换色彩，增添诗香与灵动感
- **文字不可选中** — 全局禁用用户选择，避免拖拽操作误选中界面文字，保持沉浸式体验

## 策略说明

### 1. 启发式规则 (Heuristic)

基于人工设计的棋型评分表，识别活四、冲四、活三、眠三等棋型并打分。

**优势**：快速、可解释、无需训练。执黑胜率极高 (94%)。
**劣势**：执白表现差 (27%)，依赖人工经验，上限有限。

### 2. MCTS (蒙特卡洛树搜索)

预展开候选走法，每轮模拟用 UCT 公式选择子节点，以启发式评分引导模拟结果。候选仅限已有棋子周围 2 格，大幅缩减搜索空间。

**优势**：无需训练数据，响应快（500 次模拟 < 3ms）。两面均衡（执黑 77% / 执白 60%），负场最少。
**劣势**：单层搜索，深度有限。

### 3. Alpha-Beta 博弈树搜索

迭代加深 + 11 种棋型精确识别 + 杀手启发式 + 走法排序，depth=4 中盘约 30ms。

**优势**：精确搜索，响应快，对人类玩家效果好。
**劣势**：depth=4 不足以对抗其他 AI 策略，评估函数需优化。

### 4. Minimax 剪枝搜索

Negamax + Alpha-Beta 剪枝 + 静态搜索 (Quiescence Search)，通过 Go 端管理持久 Python 子进程管道通信。depth=4，2 秒时限。

**优势**：静态搜索避免地平线效应，评估函数优于 Go 版 Alpha-Beta。
**劣势**：需 Python 子进程，执白表现差。

### 5. AlphaZero 深度强化学习

基于 ResNet 的 Policy-Value 网络 + MCTS，自对弈训练。Go 端通过持久 Python 子进程管道通信。

**优势**：自我进化，无需人类知识，棋力随训练增强。锦标赛胜率最高 (62.5%)。
**劣势**：需要 GPU 训练，推理需 Python 进程，部署较重。

## 技术栈

| 模块 | 语言 | 框架/库 |
|------|------|---------|
| 游戏引擎 | Go | 标准库 |
| 启发式策略 | Go | 标准库 |
| MCTS | Go | 标准库 |
| Alpha-Beta | Go | 标准库 |
| Minimax | Go + Python | NumPy |
| AlphaZero | Go + Python | PyTorch |
| 前端界面 | TypeScript | React + Vite |

## 许可证

MIT License
