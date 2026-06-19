#!/usr/bin/env python3
"""
AlphaZero 五子棋 — 模型定义 + MCTS + 训练逻辑。

用法:
    python alphazero_train.py                     # 从头训练
    python alphazero_train.py --resume best       # 从最优 checkpoint 恢复
    python alphazero_train.py --iters 500 --eps 200 --sims 400
"""

import os
import sys
import glob
import math
import time
import threading
import logging
import argparse
from collections import deque
from concurrent.futures import as_completed
from datetime import datetime
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# 训练专用依赖延迟导入，推理侧不进此路径
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import coloredlogs

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from gomoku import check_winner_from as _check_winner_from

EPS = 1e-8
log = logging.getLogger(__name__)

PROJECT_ROOT = _here
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
TB_DIR = os.path.join(PROJECT_ROOT, 'logs', 'tensorboard')


# ============================================================
# utils
# ============================================================

class dotdict(dict):
    """字典扩展，支持属性式访问。d.key 等价于 d['key']，提升配置可读性。"""
    def __getattr__(self, name):
        return self[name]


# ============================================================
# Game / NeuralNet 抽象基类
# ============================================================

class Game:
    """游戏环境抽象接口。

    所有方法均从"当前玩家=1"的视角操作：棋盘以 +1/-1/0 表示，玩家恒为 1。
    getCanonicalForm 负责将任意视角统一为此规范形式。
    """
    def getInitBoard(self):
        """返回初始空棋盘，shape=(rows, cols)。"""
        raise NotImplementedError
    def getBoardSize(self):
        """返回 (rows, cols)。"""
        raise NotImplementedError
    def getActionSize(self):
        """返回动作空间大小 = rows * cols。"""
        raise NotImplementedError
    def getNextState(self, board, player, action):
        """执行动作，返回 (next_board, next_player)。"""
        raise NotImplementedError
    def getValidMoves(self, board, player):
        """返回 shape=(action_size,) 的 int8 掩码，1=合法。"""
        raise NotImplementedError
    def getGameEnded(self, board, player):
        """返回 0=未结束, 1=player胜, -1=player负, 1e-4=平局。"""
        raise NotImplementedError
    def getCanonicalForm(self, board, player):
        """转换为规范形式：当前玩家=1, 对手=-1。"""
        raise NotImplementedError
    def getSymmetries(self, board, pi):
        """返回棋盘和策略的所有对称变换（8种：4旋转×2翻转），用于数据增强。"""
        raise NotImplementedError
    def stringRepresentation(self, board):
        """返回棋盘字符串表示，用于 MCTS 状态哈希。"""
        raise NotImplementedError


class NeuralNet:
    """神经网络包装器抽象接口。"""
    def train(self, examples):
        """在示例列表上训练。每个示例为 (board, pi_target, v_target)。返回平均损失。"""
        raise NotImplementedError
    def predict(self, board):
        """返回 (pi_probs_array, v_scalar)。"""
        raise NotImplementedError
    def save_checkpoint(self, folder, filename):
        raise NotImplementedError
    def load_checkpoint(self, folder, filename):
        raise NotImplementedError


# ============================================================
# Gomoku Game — 15x15 自由落子五子棋，连 5 获胜
# ============================================================

class GomokuGame(Game):
    """15×15 五子棋，自由落子（无禁手），连 5 获胜。

    棋盘编码：1=当前玩家，-1=对手，0=空。这是 AlphaZero 的标准规范形式，
    所有 Game 接口方法都基于此约定。
    """

    def __init__(self, rows=15, cols=15, connect=5):
        self.rows = rows
        self.cols = cols
        self.connect = connect
        # 预计算所有可能的五连窗口 (起点r,c + 方向dr,dc)
        # 终止判定只需检查窗口内是否全为同一玩家
        self._win_windows = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # 水平|垂直|对角线|反对角线
                    er, ec = r + (connect - 1) * dr, c + (connect - 1) * dc
                    if 0 <= er < rows and 0 <= ec < cols:
                        self._win_windows.append((r, c, dr, dc))

    def getInitBoard(self):
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def getBoardSize(self):
        return (self.rows, self.cols)

    def getActionSize(self):
        return self.rows * self.cols

    def getNextState(self, board, player, action):
        r, c = divmod(action, self.cols)
        next_board = board.copy()
        next_board[r, c] = player
        return next_board, -player

    def getValidMoves(self, board, player):
        return (board.ravel() == 0).astype(np.int8)

    def getGameEnded(self, board, player):
        """判断对局是否结束。

        通过预计算的五连窗口遍历判定，O(#windows * connect)。
        返回 1=player 胜, -1=player 负, 1e-4=平局(棋盘满), 0=未结束。
        """
        for r, c, dr, dc in self._win_windows:
            owner = board[r, c]
            if owner == 0:
                continue
            win = True
            for i in range(1, self.connect):
                if board[r + i * dr, c + i * dc] != owner:
                    win = False
                    break
            if win:
                return 1 if owner == player else -1
        # 棋盘全满且无人获胜 → 平局
        if np.all(board != 0):
            return 1e-4  # 接近 0 但不等于 0，MCTS 中用作终止符号
        return 0

    def getCanonicalForm(self, board, player):
        """转换为规范形式：当前玩家棋子 → +1，对手棋子 → -1。
        关键：乘以 player 即可，因为 player ∈ {+1, -1}。"""
        return player * board

    def getSymmetries(self, board, pi):
        """生成 8 种对称变换（D4 二面体群）：4 种旋转 × 2 种翻转。
        用于数据增强，将一局对弈样本扩展 8 倍。"""
        pi_2d = pi.reshape(self.rows, self.cols)
        results = []
        for k in range(4):             # 旋转 0°/90°/180°/270°
            for flip in [False, True]:  # 原始 / 水平翻转
                b = np.rot90(board, k)
                p = np.rot90(pi_2d, k)
                if flip:
                    b = np.fliplr(b)
                    p = np.fliplr(p)
                results.append((b.copy(), p.ravel()))
        return results

    def stringRepresentation(self, board):
        """以字节串形式序列化棋盘，用作 MCTS 状态哈希键。"""
        return board.tobytes()

    @staticmethod
    def display(board):
        n = board.shape[0]
        chars = {0: '-', 1: 'X', -1: 'O'}
        print("   " + " ".join(str(i) for i in range(n)))
        for r in range(n):
            row = " ".join(chars.get(board[r, c], '?') for c in range(n))
            print(f"{r:>2} {row}")
        print()


# ============================================================
# 神经网络 (PyTorch) — ResNet 风格策略-价值双头网络
# ============================================================

# 网络超参数：学习率、dropout、训练轮数、batch size、通道数、价值损失权重、梯度裁剪
NET_ARGS = dotdict({'lr': 0.0001, 'dropout': 0.3, 'epochs': 2,
                    'batch_size': 64, 'num_channels': 128, 'value_loss_weight': 0.5,
                    'grad_clip': 1.0})


class GomokuNNet(nn.Module):
    """AlphaZero 策略-价值网络。

    架构：共享卷积主干(4层) → FC(512→256) → 双头
    - 卷积层：128 通道，3×3 卷积，same padding，BatchNorm，ReLU
    - 全连接：512 → 256，各带 BatchNorm + Dropout(0.3)
    - 策略头：Linear(256→225) + softmax → 每格落子概率
    - 价值头：Linear(256→1) + tanh → [-1,+1] 胜率估计
    """

    def __init__(self, game, args):
        super().__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        c = args.num_channels  # 128

        # 共享卷积主干 — 4 层 conv + BN + ReLU，无池化以保留空间信息
        self.conv1 = nn.Conv2d(1, c, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, c, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(c)
        self.conv4 = nn.Conv2d(c, c, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(c)

        # 全连接层 — 展平后 128*15*15 = 28800 → 512 → 256
        fc_in = c * self.board_x * self.board_y
        self.fc1 = nn.Linear(fc_in, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(args.dropout)      # 0.3
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(args.dropout)

        # 双头输出
        self.pi_head = nn.Linear(256, self.action_size)  # 策略头 → 225 维
        self.v_head = nn.Linear(256, 1)                   # 价值头 → 1 维标量

    def forward(self, x):
        """前向传播：输入 (B, rows*cols) 或 (B, rows, cols) → 输出 (pi, v)。"""
        # 统一 reshape 为 (B, 1, rows, cols)
        x = x.view(-1, 1, self.board_x, self.board_y)
        # 卷积主干
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # 展平 → FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        # 双头：softmax 策略 + tanh 价值
        pi = F.softmax(self.pi_head(x), dim=1)
        v = torch.tanh(self.v_head(x))
        return pi, v


class NNetWrapper(NeuralNet):
    """神经网络包装器，管理 PyTorch 模型训练/推理/持久化。

    使用 torch.compile 对 CUDA 推理加速（reduce-overhead 模式），
    首推时触发 JIT 编译，后续调用享受融合算子加速。
    """

    def __init__(self, game, device=None):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet = GomokuNNet(game, NET_ARGS).to(self.device)
        self.optimizer = Adam(self.nnet.parameters(), lr=NET_ARGS.lr)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True       # 自动搜索最优卷积算法
            torch.set_float32_matmul_precision('high')  # TensorFloat-32 加速
            # torch.compile 需要 eval 模式下的固定输入 shape
            self.nnet.eval()
            self._nnet_infer = torch.compile(self.nnet, mode='reduce-overhead')
        else:
            self._nnet_infer = self.nnet

    def train(self, examples):
        """在训练样本上执行监督学习。

        examples: [(board, pi_target, v_target), ...]
        损失 = CrossEntropy(policy) + 0.5 * MSE(value)
        梯度裁剪防止梯度爆炸，Adam 优化器更新。
        返回所有 epoch 的平均损失。
        """
        boards, pis, vs = list(zip(*examples))
        boards = torch.FloatTensor(np.array(boards)).to(self.device)
        target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
        target_vs = torch.FloatTensor(np.array(vs)).to(self.device).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(boards, target_pis, target_vs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=NET_ARGS.batch_size, shuffle=True)

        self.nnet.train()
        epoch_losses = []
        for _ in range(NET_ARGS.epochs):
            batch_losses = []
            for batch_boards, batch_pis, batch_vs in loader:
                pred_pis, pred_vs = self.nnet(batch_boards)
                # 策略损失：交叉熵 = -sum(target * log(pred))
                loss_pi = -torch.sum(batch_pis * torch.log(pred_pis + 1e-8)) / batch_pis.size(0)
                # 价值损失：均方误差
                loss_v = F.mse_loss(pred_vs, batch_vs)
                # 总损失：策略损失 + 价值权重 * 价值损失
                loss = loss_pi + NET_ARGS.value_loss_weight * loss_v
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), NET_ARGS.grad_clip)
                self.optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        return sum(epoch_losses) / len(epoch_losses)

    def predict(self, board):
        """单棋盘推理。board shape=(rows, cols)，返回 (pi_array, v_scalar)。"""
        self.nnet.eval()
        with torch.inference_mode():  # 禁用 autograd，减少显存
            x = torch.as_tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
            pi, v = self._nnet_infer(x)
            return pi.cpu().numpy()[0], v.cpu().numpy()[0, 0]

    def save_checkpoint(self, folder, filename):
        """保存模型权重 + 优化器状态到 .pth.tar 文件。"""
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='', filename='best.pth.tar'):
        """加载模型权重 + 优化器状态。加载后重新编译推理图。"""
        filepath = os.path.join(folder, filename) if folder else filename
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.device.type == 'cuda':
            self.nnet.eval()
            self._nnet_infer = torch.compile(self.nnet, mode='reduce-overhead')


# ============================================================
# MCTS — 蒙特卡洛树搜索 (AlphaZero 风格)
# ============================================================
#
# 核心公式：
#   PUCT: U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
#   其中 Q=平均价值, P=先验概率(来自网络), N=访问次数
#
# 搜索分四阶段：
#   1. SELECT — 从根沿 PUCT 最大分支走到叶子
#   2. EXPAND  — 用网络评估叶子，初始化 Q/P/N
#   3. BACKUP  — 将叶子价值反传更新路径上所有节点
#   4. 最终走法 — 选访问次数最多的分支（温度控制探索）
#
# 状态索引：所有缓存字典以 stringRepresentation(board) 为键
# 价值约定：+1=当前玩家胜, -1=当前玩家负，反传时取反切换视角

class MCTS:
    def __init__(self, game, nnet, args, predict_fn=None):
        self.game = game
        self.nnet = nnet
        # 支持外部注入 predict_fn（并行 worker 场景）
        self.predict_fn = predict_fn if predict_fn is not None else nnet.predict
        self.args = args
        self.action_size = game.getActionSize()
        # 以棋盘字节串为键的缓存字典
        self._Qsa = {}   # Q(s,a): 状态-动作平均价值
        self._Nsa = {}   # N(s,a): 状态-动作访问次数
        self.Ns = {}     # N(s):   状态总访问次数
        self.Ps = {}     # P(s):   先验概率（网络输出 + 狄利克雷噪声）
        self.Es = {}     # E(s):   终局状态（0=未结束, ±1=胜负, 1e-4=平）
        self.Vs = {}     # V(s):   合法走法掩码

    def reset(self):
        """清空搜索树，每局对弈开始时调用。"""
        self._Qsa.clear()
        self._Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Es.clear()
        self.Vs.clear()

    def getActionProb(self, canonicalBoard, temp=1, add_noise=False):
        """执行 numMCTSSims 次搜索后返回动作概率分布。

        temp=0: 确定性选择（选访问次数最多的走法）
        temp>0: 按访问次数^(1/temp) 采样，temp=1 为标准探索温度
        add_noise=True: 在根节点加入狄利克雷噪声，增加对弈多样性
        """
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard, add_noise=add_noise)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = self._Nsa.get(s, np.zeros(self.action_size, dtype=np.int32))

        # 温度=0 → 贪心：随机打破平局
        if temp == 0:
            best = np.flatnonzero(counts == np.max(counts))
            probs = np.zeros(self.action_size)
            probs[np.random.choice(best)] = 1
            return probs

        # 温度缩放：counts^(1/temp)，然后归一化
        counts = counts.astype(np.float64) ** (1.0 / temp)
        return counts / counts.sum()

    def search(self, canonicalBoard, add_noise=False):
        """递归执行一次 MCTS 搜索（SELECT → EXPAND → BACKUP）。

        返回 -v：从当前玩家视角取反的价值。
        递归深度 = 对局剩余步数，树沿合法分支自然终止。
        """
        s = self.game.stringRepresentation(canonicalBoard)

        # ── 终止检查 ──
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]  # 终局，返回取反结果

        # ── 叶子节点：EXPAND ──
        if s not in self.Ps:
            # 网络评估
            self.Ps[s], v = self.predict_fn(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)

            # 掩码：将非法走法概率置零，重新归一化
            self.Ps[s] = self.Ps[s] * valids
            s_sum = np.sum(self.Ps[s])
            if s_sum > 0:
                self.Ps[s] /= s_sum
            else:
                # 极端情况：网络对所有合法走法输出 ≈0。均匀分配避免除零。
                log.error("所有合法走法被屏蔽")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            # 初始化统计量
            self.Vs[s] = valids
            self.Ns[s] = 0
            self._Qsa[s] = np.zeros(self.action_size, dtype=np.float32)
            self._Nsa[s] = np.zeros(self.action_size, dtype=np.int32)

            # 根节点加入狄利克雷噪声 → 探索
            if add_noise:
                valid_mask = valids.astype(bool)
                n_valid = valid_mask.sum()
                noise = np.random.dirichlet([self.args.dirichlet_alpha] * n_valid)
                self.Ps[s][valid_mask] = (1 - self.args.dirichlet_epsilon) * self.Ps[s][valid_mask] + self.args.dirichlet_epsilon * noise

            return -v

        # ── 内部节点：SELECT ──
        valids = self.Vs[s]
        # PUCT 公式：Q + cpuct * P * sqrt(N_parent) / (1 + N_child)
        ns_sqrt = math.sqrt(self.Ns[s])
        u_arr = self._Qsa[s] + self.args.cpuct * self.Ps[s] * ns_sqrt / (1.0 + self._Nsa[s])
        u_arr[valids == 0] = -np.inf  # 屏蔽非法走法
        a = int(np.argmax(u_arr))

        # 执行动作并切换视角（取反玩家 + 转规范形式）
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        # 递归搜索
        v = self.search(next_s)

        # ── BACKUP：增量更新 Q = 累加平均 ──
        n = self._Nsa[s][a]
        self._Qsa[s][a] = (n * self._Qsa[s][a] + v) / (n + 1)
        self._Nsa[s][a] = n + 1
        self.Ns[s] += 1
        return -v


# ============================================================
# AlphaZero Player — 对外推理接口
# ============================================================

def _env_board_to_canonical(board, player):
    """Go 格式棋盘 (0=空,1=黑,2=白) → AlphaZero 规范形式 (+1/-1/0)。
    当前玩家棋子→+1，对手棋子→-1。"""
    az_board = np.where(board == 0, 0, np.where(board == player, 1, -1)).astype(np.int8)
    return az_board


class AlphaZeroPlayer:
    """AlphaZero 玩家，封装模型加载 + MCTS 推理。

    供外部（评估/对战）调用，不与训练循环耦合。
    使用方法：
        player = AlphaZeroPlayer('best.pth.tar')
        action = player.predict_first(obs)          # 首步
        action = player.opponent_callback(board, p) # 后续回合
    """

    def __init__(self, checkpoint_path=None, numMCTSSims=800, cpuct=1.0):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best.pth.tar')

        self.game = GomokuGame()
        self.nnet = NNetWrapper(self.game)
        self.nnet.load_checkpoint(filename=checkpoint_path)
        self.args = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': cpuct})
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self._name = f"az_mcts{numMCTSSims}"

    @property
    def name(self):
        return self._name

    def reset(self):
        """重置 MCTS 树，新对局开始时调用。"""
        self.mcts.reset()

    def predict_first(self, obs):
        """从环境观测预测第一步。obs 为 Go 格式的展平棋盘 (225,)。"""
        n_cells = self.game.rows * self.game.cols
        board_flat = (obs[:n_cells] * 2.0).astype(np.int8)
        board = board_flat.reshape(self.game.rows, self.game.cols)
        canonical = _env_board_to_canonical(board, 1)
        pi = self.mcts.getActionProb(canonical, temp=0)
        return int(np.argmax(pi))

    def opponent_callback(self, board, player):
        """对手落子后的回调，返回 AI 的应对走法。"""
        canonical = _env_board_to_canonical(board, player)
        pi = self.mcts.getActionProb(canonical, temp=0)
        return int(np.argmax(pi))


# ============================================================
# Arena — 竞技场评估：新模型 vs 旧模型
# ============================================================

class Arena:
    """两玩家对弈竞技场。

    执行 num 局对弈，先 player1 执先手(num/2 局)，
    再交换先后手(player2 执先手 num/2 局)，消除先手偏差。
    """

    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """单局对弈。players 数组索引技巧：
        curPlayer ∈ {+1, -1}，players[curPlayer+1] 在 +1→players[2]=p1, -1→players[0]=p2
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1  # 始终从 +1 方开始
        board = self.game.getInitBoard()

        while self.game.getGameEnded(board, curPlayer) == 0:
            if verbose and self.display:
                self.display(board)
            canonical = self.game.getCanonicalForm(board, curPlayer)
            action = players[curPlayer + 1](canonical)
            valids = self.game.getValidMoves(canonical, 1)
            if valids[action] == 0:
                log.error(f'走法 {action} 无效！')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        if verbose and self.display:
            self.display(board)
        # 返回值：+1=player1 胜, -1=player2 胜, ~0=平
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """批量对弈，两半场互换先后手以消除先手优势。"""
        from tqdm import tqdm
        half = num // 2
        oneWon, twoWon, draws = 0, 0, 0

        # 上半场：player1 执先手
        for _ in tqdm(range(half), desc="Arena (1)"):
            result = self.playGame(verbose=verbose)
            if result == 1:
                oneWon += 1
            elif result == -1:
                twoWon += 1
            else:
                draws += 1

        # 交换先后手
        self.player1, self.player2 = self.player2, self.player1

        # 下半场：player2 执先手（现在是 player1）
        for _ in tqdm(range(half), desc="Arena (2)"):
            result = self.playGame(verbose=verbose)
            if result == -1:
                oneWon += 1  # 注意：此时胜负指示反转
            elif result == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws


# ============================================================
# Coach — AlphaZero 训练循环：自对弈 + 学习 + 评估
# ============================================================
#
# 每轮迭代流程：
#   1. 自对弈 (self-play):   当前网络 + MCTS 对弈 numEps 局，生成训练样本
#   2. 训练 (train):         在累积样本上训练，输出候选模型
#   3. 评估 (arena):         候选 vs 当前最优，>55% 胜率即接受
#   4. 模型管理:             接受 → 替换 best.pth.tar；拒绝 → 回滚权重
#
# 样本格式: (canonical_board, pi_mcts_target, game_result)
#   其中 pi_mcts_target 是 MCTS 访问计数分布（软标签），game_result ∈ {-1,0,+1}

class Coach:
    def __init__(self, game, nnet, args):
        from torch.utils.tensorboard import SummaryWriter
        self.game = game
        self.nnet = nnet              # 当前训练中的网络
        self.pnet = nnet.__class__(self.game)  # 上一版本网络（竞技场对手）
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # 滑动窗口：保留最近 N 轮的训练样本
        self.skipFirstSelfPlay = False  # 续训标志：跳过已完成的首次自对弈
        self.writer = SummaryWriter(log_dir=TB_DIR)

    @staticmethod
    def _run_episode(game, args, predict_fn):
        """执行一局自对弈，返回训练样本列表。

        每步：
        1. MCTS 搜索 → 访问计数分布 pi（加入狄利克雷噪声探索）
        2. 按 pi 采样动作并执行
        3. 记录 (棋盘, 当前玩家, pi) 三元组
        4. 应用 8 种对称变换扩展样本
        对局结束后，根据结果标注每个样本的价值。
        """
        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        # 注意：mcts 的 nnet 参数为 None，仅使用 predict_fn
        mcts = MCTS(game, None, args, predict_fn=predict_fn)

        while True:
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            # 前 tempThreshold 步用温度=1（探索），后续用温度=0（贪心）
            temp = int(len(trainExamples) < args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp, add_noise=True)
            # 数据增强：8 种对称变换
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append((b, curPlayer, p))

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)

            r = game.getGameEnded(board, curPlayer)
            if r != 0:
                # 结果标注：从当前局面玩家视角赋予价值
                # ((-1) ** (player != curPlayer)) 将全局结果映射到每个样本的玩家视角
                return [(b, p, r * ((-1) ** (player != curPlayer))) for b, player, p in trainExamples]

    def _self_play_sequential(self):
        """串行自对弈（单进程模式）。"""
        from tqdm import tqdm
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for _ in tqdm(range(self.args.numEps), desc="Self Play"):
            iterationTrainExamples += Coach._run_episode(self.game, self.args, self.nnet.predict)
        return iterationTrainExamples

    def _self_play_parallel(self):
        """并行自对弈，使用 ProcessPoolExecutor 多进程加速。

        关键设计：
        - 用 spawn 上下文避免 CUDA fork 问题
        - 网络权重通过 state_dict 序列化传递给 worker
        - 每个 worker 独立持有网络副本 + MCTS 实例
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm

        n_workers = min(self.args.num_workers, self.args.numEps, os.cpu_count() or 4)
        # 将权重复制到 CPU 再序列化，避免 GPU 张量跨进程传输
        state_dict = {k: v.cpu().clone() for k, v in self.nnet.nnet.state_dict().items()}
        game_args = (self.game.rows, self.game.cols, self.game.connect)
        args_dict = dict(self.args)

        ctx = mp.get_context('spawn')
        examples = deque([], maxlen=self.args.maxlenOfQueue)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_self_play_worker,  # 每个 worker 启动时初始化
            initargs=(game_args, args_dict, state_dict)
        ) as pool:
            futures = [pool.submit(_run_self_play_episode, None)
                       for _ in range(self.args.numEps)]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Self Play"):
                examples += f.result()

        return examples

    def _train_and_evaluate(self, trainExamples, i):
        """训练 + 竞技场评估一步。

        1. 保存当前权重为 temp
        2. 在训练样本上训练
        3. 并行竞技场：新模型 vs temp（旧模型）
        4. 根据阈值决定接受/拒绝
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        # 保存快照供竞技场使用
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

        train_loss = self.nnet.train(trainExamples)

        log.info('与上一版本对弈中...')

        # 准备并行竞技场：序列化新旧两版权重
        new_state = {k: v.cpu().clone() for k, v in self.nnet.nnet.state_dict().items()}
        old_state = {k: v.cpu().clone() for k, v in self.pnet.nnet.state_dict().items()}
        game_args = (self.game.rows, self.game.cols, self.game.connect)

        n_games = self.args.arenaCompare
        half = n_games // 2
        # 任务列表：前半 agent 新模型先手，后半旧模型先手
        tasks = [(True, i) for i in range(half + n_games % 2)] + [(False, i) for i in range(half)]

        nwins = pwins = 0
        draws = 0

        n_arena_workers = min(4, n_games, os.cpu_count() or 4)
        ctx = mp.get_context('spawn')

        with ProcessPoolExecutor(
            max_workers=n_arena_workers,
            mp_context=ctx,
            initializer=_init_arena_worker,
            initargs=(game_args, new_state, old_state, dict(self.args))
        ) as pool:
            futures = [pool.submit(_arena_play_game, new_first) for new_first, _ in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Arena"):
                result = f.result()
                if result == 1:
                    nwins += 1
                elif result == -1:
                    pwins += 1
                else:
                    draws += 1

        # 胜率 = 新模型胜 / (新+旧胜场)，平局不计入
        win_rate = float(nwins) / (pwins + nwins) if pwins + nwins > 0 else 0.5

        log.info(f'新版/旧版 胜场: {nwins} / {pwins} ; 平局: {draws}')

        # TensorBoard 记录
        self.writer.add_scalar('Train/Loss', train_loss, i)
        self.writer.add_scalar('Arena/NewWins', nwins, i)
        self.writer.add_scalar('Arena/PrevWins', pwins, i)
        self.writer.add_scalar('Arena/Draws', draws, i)
        self.writer.add_scalar('Arena/WinRate', win_rate, i)
        self.writer.add_scalar('Train/ExamplesCount', len(trainExamples), i)

        # 模型接受/拒绝逻辑
        if pwins + nwins == 0 or win_rate < self.args.updateThreshold:
            log.info('拒绝新模型')
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')  # 回滚
            self.writer.add_scalar('Arena/Accepted', 0, i)
        else:
            log.info('接受新模型')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'checkpoint_{i}.pth.tar')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            self.writer.add_scalar('Arena/Accepted', 1, i)

    def learn(self):
        """主训练循环：迭代 numIters 轮，每轮执行 自对弈→训练→评估。

        学习率按余弦退火从 lr 衰减到 lr*0.1。
        保留最近 numItersForTrainExamplesHistory 轮的样本以避免遗忘。
        """
        # 余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.nnet.optimizer, T_max=self.args.numIters, eta_min=NET_ARGS.lr * 0.1)

        for i in range(1, self.args.numIters + 1):
            log.info(f'开始第 {i} 轮迭代 ...')

            # 自对弈生成样本
            if not self.skipFirstSelfPlay or i > 1:
                if self.args.num_workers > 1:
                    self.trainExamplesHistory.append(self._self_play_parallel())
                else:
                    self.trainExamplesHistory.append(self._self_play_sequential())

            # 滑动窗口：仅保留最近 N 轮样本
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning("移除最早的训练样本")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)

            # 展平所有历史样本并打乱
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self._train_and_evaluate(trainExamples, i)
            scheduler.step()

        self.writer.close()

    def saveTrainExamples(self, iteration):
        """将训练样本历史持久化到 .examples 文件。"""
        folder = self.args.checkpoint
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f'checkpoint_{iteration}.pth.tar.examples')
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        """从 .examples 文件恢复训练样本历史（续训时调用）。"""
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'训练样本文件 "{examplesFile}" 未找到！')
        else:
            log.info("从文件加载训练样本...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('加载完成！')
            self.skipFirstSelfPlay = True  # 续训：跳过已完成的首次自对弈


# ============================================================
# 多进程 Worker 函数（模块级，供 ProcessPoolExecutor 初始化）
# ============================================================
#
# Python 的 ProcessPoolExecutor 要求 worker 初始化函数和任务函数
# 都必须是模块级可 pickle 的对象（不能是类方法或 lambda）。
# 因此以下函数均为模块级，通过 global 变量在 worker 进程内共享状态。

# ── 自对弈 Worker ──

_worker_game = None         # worker 进程内的游戏实例
_worker_args = None         # worker 进程内的配置
_worker_predict_fn = None   # worker 进程内的网络推理函数


def _init_self_play_worker(game_args, args_dict, state_dict):
    """自对弈 worker 初始化：加载网络权重并预热推理。
    预热调用确保 CUDA 上下文和 torch.compile 在真实任务前完成。"""
    global _worker_game, _worker_args, _worker_predict_fn
    _worker_game = GomokuGame(*game_args)
    nnet = NNetWrapper(_worker_game)
    nnet.nnet.load_state_dict(state_dict)
    # 预热：触发 JIT 编译和 CUDA 内核初始化
    dummy = _worker_game.getInitBoard()
    nnet.predict(dummy)
    _worker_args = dotdict(args_dict)
    _worker_predict_fn = nnet.predict


def _run_self_play_episode(_unused):
    """执行一局自对弈，返回训练样本。参数 _unused 满足 pool.submit 签名。"""
    global _worker_game, _worker_args, _worker_predict_fn
    return Coach._run_episode(_worker_game, _worker_args, _worker_predict_fn)


# ── 竞技场 Worker ──

_arena_game = None          # worker 进程内的游戏实例
_arena_args = None          # worker 进程内的配置
_arena_nnet_new = None      # 新模型网络
_arena_nnet_old = None      # 旧模型网络


def _init_arena_worker(game_args, new_state, old_state, args_dict):
    """竞技场 worker 初始化：加载新旧两个模型的权重并预热。"""
    global _arena_game, _arena_args, _arena_nnet_new, _arena_nnet_old
    _arena_game = GomokuGame(*game_args)
    _arena_args = dotdict(args_dict)

    _arena_nnet_new = NNetWrapper(_arena_game)
    _arena_nnet_new.nnet.load_state_dict(new_state)
    _arena_nnet_old = NNetWrapper(_arena_game)
    _arena_nnet_old.nnet.load_state_dict(old_state)

    # 预热两个网络
    dummy = _arena_game.getInitBoard()
    _arena_nnet_new.predict(dummy)
    _arena_nnet_old.predict(dummy)


def _arena_play_game(new_first):
    """执行一局竞技场对弈。

    new_first=True: 新模型执先手
    返回 +1=新模型胜, -1=旧模型胜, 0=平局
    """
    global _arena_game, _arena_args, _arena_nnet_new, _arena_nnet_old

    # 为每局创建独立的 MCTS 实例（清空搜索树）
    mcts_new = MCTS(_arena_game, _arena_nnet_new, _arena_args)
    mcts_old = MCTS(_arena_game, _arena_nnet_old, _arena_args)

    if new_first:
        first_mcts, second_mcts = mcts_new, mcts_old
    else:
        first_mcts, second_mcts = mcts_old, mcts_new

    # players 索引技巧同 Arena.playGame
    players = [second_mcts, None, first_mcts]
    curPlayer = 1
    board = _arena_game.getInitBoard()

    while _arena_game.getGameEnded(board, curPlayer) == 0:
        canonical = _arena_game.getCanonicalForm(board, curPlayer)
        # 竞技场中使用温度=0（确定性选择）
        action = int(np.argmax(players[curPlayer + 1].getActionProb(canonical, temp=0)))
        valids = _arena_game.getValidMoves(canonical, 1)
        # 极端回退：如果 MCTS 选出非法走法（理论上不应发生），随机选合法走法
        if valids[action] == 0:
            action = int(np.random.choice(np.flatnonzero(valids)))
        board, curPlayer = _arena_game.getNextState(board, curPlayer, action)

    game_result = curPlayer * _arena_game.getGameEnded(board, curPlayer)

    # 将结果映射回"新模型 vs 旧模型"的胜负
    if game_result > 0.5:
        return 1 if new_first else -1
    elif game_result < -0.5:
        return -1 if new_first else 1
    return 0


# ============================================================
# 训练入口 — 参数配置 + 主函数
# ============================================================

# 默认训练配置 — 按 AlphaZero 论文建议值设定
DEFAULT_ARGS = dotdict({
    'numIters': 200,              # 总迭代轮数
    'numEps': 200,                # 每轮自对弈局数
    'tempThreshold': 15,          # 前 N 步使用温度=1 探索，之后温度=0 贪心
    'updateThreshold': 0.55,      # 竞技场胜率阈值：新模型需 ≥55% 才接受
    'maxlenOfQueue': 100000,      # 训练样本队列最大长度
    'numMCTSSims': 800,           # 每步 MCTS 模拟次数
    'arenaCompare': 40,           # 竞技场评估局数
    'cpuct': 1.0,                 # PUCT 探索常数（控制搜索宽度）
    'dirichlet_alpha': 1.0,       # 狄利克雷噪声浓度参数（≈1/n_valid 时接近均匀）
    'dirichlet_epsilon': 0.25,    # 噪声混入比例：P' = (1-ε)*P + ε*noise
    'num_workers': 8,             # 并行自对弈 worker 数
    'checkpoint': CHECKPOINT_DIR,
    'load_model': False,
    'load_folder_file': (CHECKPOINT_DIR, 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,  # 保留最近 N 轮样本（滑动窗口）
})


def setup_file_logging():
    """配置文件日志：同时输出到控制台（coloredlogs）和日志文件。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f'train_{timestamp}.log')

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logging.getLogger(__name__).addHandler(file_handler)

    log.info(f'日志输出至 {log_path}')
    return log_path


def main():
    import coloredlogs
    parser = argparse.ArgumentParser(description='AlphaZero Gomoku Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint name (e.g. "best", "checkpoint_10")')
    parser.add_argument('--iters', type=int, default=None, help='Override numIters')
    parser.add_argument('--eps', type=int, default=None, help='Override numEps per iteration')
    parser.add_argument('--sims', type=int, default=None, help='Override numMCTSSims')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel self-play workers (default 1, set 4-8 for GPU)')
    opts = parser.parse_args()

    coloredlogs.install(level='INFO')
    log_path = setup_file_logging()

    # 合并命令行参数覆盖默认值
    args = dotdict({**DEFAULT_ARGS})
    if opts.iters is not None:
        args['numIters'] = opts.iters
    if opts.eps is not None:
        args['numEps'] = opts.eps
    if opts.sims is not None:
        args['numMCTSSims'] = opts.sims
    if opts.workers is not None:
        args['num_workers'] = opts.workers

    log.info('加载五子棋游戏 (15×15, 五连获胜)...')
    game = GomokuGame(rows=15, cols=15, connect=5)

    log.info('加载 PyTorch 神经网络...')
    nnet = NNetWrapper(game)

    if opts.resume:
        checkpoint_name = opts.resume
        if not checkpoint_name.endswith('.pth.tar'):
            checkpoint_name = f'{checkpoint_name}.pth.tar'
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        if os.path.exists(checkpoint_path):
            log.info(f'从 {checkpoint_path} 加载模型检查点')
            nnet.load_checkpoint(filename=checkpoint_path)
            args['load_model'] = True
            args['load_folder_file'] = (CHECKPOINT_DIR, checkpoint_name)
        else:
            log.error(f'模型检查点未找到: {checkpoint_path}')
            sys.exit(1)
    else:
        log.warning('未加载检查点 — 从头开始训练')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log.info(f'模型保存至 {CHECKPOINT_DIR}')
    log.info(f'日志保存至 {log_path}')
    log.info(f'TensorBoard: tensorboard --logdir {TB_DIR}')
    log.info(f'配置: numIters={args.numIters}, numEps={args.numEps}, '
             f'numMCTSSims={args.numMCTSSims}, arenaCompare={args.arenaCompare}, '
             f'workers={args.num_workers}')

    log.info('启动训练管理器...')
    coach = Coach(game, nnet, args)

    if args.load_model:
        log.info("从文件加载训练样本...")
        coach.loadTrainExamples()

    log.info('开始 AlphaZero 训练循环')
    coach.learn()


if __name__ == '__main__':
    main()
