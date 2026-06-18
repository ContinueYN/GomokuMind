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
    def __getattr__(self, name):
        return self[name]


# ============================================================
# Game / NeuralNet 抽象基类
# ============================================================

class Game:
    def getInitBoard(self):
        raise NotImplementedError
    def getBoardSize(self):
        raise NotImplementedError
    def getActionSize(self):
        raise NotImplementedError
    def getNextState(self, board, player, action):
        raise NotImplementedError
    def getValidMoves(self, board, player):
        raise NotImplementedError
    def getGameEnded(self, board, player):
        raise NotImplementedError
    def getCanonicalForm(self, board, player):
        raise NotImplementedError
    def getSymmetries(self, board, pi):
        raise NotImplementedError
    def stringRepresentation(self, board):
        raise NotImplementedError


class NeuralNet:
    def train(self, examples):
        raise NotImplementedError
    def predict(self, board):
        raise NotImplementedError
    def save_checkpoint(self, folder, filename):
        raise NotImplementedError
    def load_checkpoint(self, folder, filename):
        raise NotImplementedError


# ============================================================
# Gomoku Game — 15x15 自由落子五子棋，连 5 获胜
# ============================================================

class GomokuGame(Game):
    def __init__(self, rows=15, cols=15, connect=5):
        self.rows = rows
        self.cols = cols
        self.connect = connect
        self._win_windows = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
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
        if np.all(board != 0):
            return 1e-4
        return 0

    def getCanonicalForm(self, board, player):
        return player * board

    def getSymmetries(self, board, pi):
        pi_2d = pi.reshape(self.rows, self.cols)
        results = []
        for k in range(4):
            for flip in [False, True]:
                b = np.rot90(board, k)
                p = np.rot90(pi_2d, k)
                if flip:
                    b = np.fliplr(b)
                    p = np.fliplr(p)
                results.append((b.copy(), p.ravel()))
        return results

    def stringRepresentation(self, board):
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
# 神经网络 (PyTorch)
# ============================================================

NET_ARGS = dotdict({'lr': 0.0001, 'dropout': 0.3, 'epochs': 2,
                    'batch_size': 64, 'num_channels': 128, 'value_loss_weight': 0.5,
                    'grad_clip': 1.0})


class GomokuNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        c = args.num_channels

        self.conv1 = nn.Conv2d(1, c, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, c, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(c)
        self.conv4 = nn.Conv2d(c, c, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(c)

        fc_in = c * self.board_x * self.board_y
        self.fc1 = nn.Linear(fc_in, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(args.dropout)

        self.pi_head = nn.Linear(256, self.action_size)
        self.v_head = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.board_x, self.board_y)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        pi = F.softmax(self.pi_head(x), dim=1)
        v = torch.tanh(self.v_head(x))
        return pi, v


class NNetWrapper(NeuralNet):
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
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            self.nnet.eval()
            self._nnet_infer = torch.compile(self.nnet, mode='reduce-overhead')
        else:
            self._nnet_infer = self.nnet

    def train(self, examples):
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
                loss_pi = -torch.sum(batch_pis * torch.log(pred_pis + 1e-8)) / batch_pis.size(0)
                loss_v = F.mse_loss(pred_vs, batch_vs)
                loss = loss_pi + NET_ARGS.value_loss_weight * loss_v
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), NET_ARGS.grad_clip)
                self.optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        return sum(epoch_losses) / len(epoch_losses)

    def predict(self, board):
        self.nnet.eval()
        with torch.inference_mode():
            x = torch.as_tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
            pi, v = self._nnet_infer(x)
            return pi.cpu().numpy()[0], v.cpu().numpy()[0, 0]

    def save_checkpoint(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='', filename='best.pth.tar'):
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
# MCTS
# ============================================================

class MCTS:
    def __init__(self, game, nnet, args, predict_fn=None):
        self.game = game
        self.nnet = nnet
        self.predict_fn = predict_fn if predict_fn is not None else nnet.predict
        self.args = args
        self.action_size = game.getActionSize()
        self._Qsa = {}
        self._Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def reset(self):
        self._Qsa.clear()
        self._Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Es.clear()
        self.Vs.clear()

    def getActionProb(self, canonicalBoard, temp=1, add_noise=False):
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard, add_noise=add_noise)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = self._Nsa.get(s, np.zeros(self.action_size, dtype=np.int32))

        if temp == 0:
            best = np.flatnonzero(counts == np.max(counts))
            probs = np.zeros(self.action_size)
            probs[np.random.choice(best)] = 1
            return probs

        counts = counts.astype(np.float64) ** (1.0 / temp)
        return counts / counts.sum()

    def search(self, canonicalBoard, add_noise=False):
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            self.Ps[s], v = self.predict_fn(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids
            s_sum = np.sum(self.Ps[s])
            if s_sum > 0:
                self.Ps[s] /= s_sum
            else:
                log.error("所有合法走法被屏蔽")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            self._Qsa[s] = np.zeros(self.action_size, dtype=np.float32)
            self._Nsa[s] = np.zeros(self.action_size, dtype=np.int32)

            if add_noise:
                valid_mask = valids.astype(bool)
                n_valid = valid_mask.sum()
                noise = np.random.dirichlet([self.args.dirichlet_alpha] * n_valid)
                self.Ps[s][valid_mask] = (1 - self.args.dirichlet_epsilon) * self.Ps[s][valid_mask] + self.args.dirichlet_epsilon * noise

            return -v

        valids = self.Vs[s]
        ns_sqrt = math.sqrt(self.Ns[s])
        u_arr = self._Qsa[s] + self.args.cpuct * self.Ps[s] * ns_sqrt / (1.0 + self._Nsa[s])
        u_arr[valids == 0] = -np.inf
        a = int(np.argmax(u_arr))

        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        n = self._Nsa[s][a]
        self._Qsa[s][a] = (n * self._Qsa[s][a] + v) / (n + 1)
        self._Nsa[s][a] = n + 1
        self.Ns[s] += 1
        return -v


# ============================================================
# AlphaZero Player — 推理接口
# ============================================================

def _env_board_to_canonical(board, player):
    az_board = np.where(board == 0, 0, np.where(board == player, 1, -1)).astype(np.int8)
    return az_board


class AlphaZeroPlayer:
    """AlphaZero 玩家，兼容推理端调用接口。"""

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
        self.mcts.reset()

    def predict_first(self, obs):
        n_cells = self.game.rows * self.game.cols
        board_flat = (obs[:n_cells] * 2.0).astype(np.int8)
        board = board_flat.reshape(self.game.rows, self.game.cols)
        canonical = _env_board_to_canonical(board, 1)
        pi = self.mcts.getActionProb(canonical, temp=0)
        return int(np.argmax(pi))

    def opponent_callback(self, board, player):
        canonical = _env_board_to_canonical(board, player)
        pi = self.mcts.getActionProb(canonical, temp=0)
        return int(np.argmax(pi))


# ============================================================
# Arena
# ============================================================

class Arena:
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        players = [self.player2, None, self.player1]
        curPlayer = 1
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
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        from tqdm import tqdm
        half = num // 2
        oneWon, twoWon, draws = 0, 0, 0

        for _ in tqdm(range(half), desc="Arena (1)"):
            result = self.playGame(verbose=verbose)
            if result == 1:
                oneWon += 1
            elif result == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(half), desc="Arena (2)"):
            result = self.playGame(verbose=verbose)
            if result == -1:
                oneWon += 1
            elif result == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws


# ============================================================
# Coach — 自对弈 + 学习循环
# ============================================================

class Coach:
    def __init__(self, game, nnet, args):
        from torch.utils.tensorboard import SummaryWriter
        self.game = game
        self.nnet = nnet
        self.pnet = nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        self.writer = SummaryWriter(log_dir=TB_DIR)

    @staticmethod
    def _run_episode(game, args, predict_fn):
        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        mcts = MCTS(game, None, args, predict_fn=predict_fn)

        while True:
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            temp = int(len(trainExamples) < args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp, add_noise=True)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append((b, curPlayer, p))

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)

            r = game.getGameEnded(board, curPlayer)
            if r != 0:
                return [(b, p, r * ((-1) ** (player != curPlayer))) for b, player, p in trainExamples]

    def _self_play_sequential(self):
        from tqdm import tqdm
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for _ in tqdm(range(self.args.numEps), desc="Self Play"):
            iterationTrainExamples += Coach._run_episode(self.game, self.args, self.nnet.predict)
        return iterationTrainExamples

    def _self_play_parallel(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm

        n_workers = min(self.args.num_workers, self.args.numEps, os.cpu_count() or 4)
        state_dict = {k: v.cpu().clone() for k, v in self.nnet.nnet.state_dict().items()}
        game_args = (self.game.rows, self.game.cols, self.game.connect)
        args_dict = dict(self.args)

        ctx = mp.get_context('spawn')
        examples = deque([], maxlen=self.args.maxlenOfQueue)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_self_play_worker,
            initargs=(game_args, args_dict, state_dict)
        ) as pool:
            futures = [pool.submit(_run_self_play_episode, None)
                       for _ in range(self.args.numEps)]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Self Play"):
                examples += f.result()

        return examples

    def _train_and_evaluate(self, trainExamples, i):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

        train_loss = self.nnet.train(trainExamples)

        log.info('与上一版本对弈中...')

        new_state = {k: v.cpu().clone() for k, v in self.nnet.nnet.state_dict().items()}
        old_state = {k: v.cpu().clone() for k, v in self.pnet.nnet.state_dict().items()}
        game_args = (self.game.rows, self.game.cols, self.game.connect)

        n_games = self.args.arenaCompare
        half = n_games // 2
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

        win_rate = float(nwins) / (pwins + nwins) if pwins + nwins > 0 else 0.5

        log.info(f'新版/旧版 胜场: {nwins} / {pwins} ; 平局: {draws}')

        self.writer.add_scalar('Train/Loss', train_loss, i)
        self.writer.add_scalar('Arena/NewWins', nwins, i)
        self.writer.add_scalar('Arena/PrevWins', pwins, i)
        self.writer.add_scalar('Arena/Draws', draws, i)
        self.writer.add_scalar('Arena/WinRate', win_rate, i)
        self.writer.add_scalar('Train/ExamplesCount', len(trainExamples), i)

        if pwins + nwins == 0 or win_rate < self.args.updateThreshold:
            log.info('拒绝新模型')
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.writer.add_scalar('Arena/Accepted', 0, i)
        else:
            log.info('接受新模型')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'checkpoint_{i}.pth.tar')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            self.writer.add_scalar('Arena/Accepted', 1, i)

    def learn(self):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.nnet.optimizer, T_max=self.args.numIters, eta_min=NET_ARGS.lr * 0.1)

        for i in range(1, self.args.numIters + 1):
            log.info(f'开始第 {i} 轮迭代 ...')

            if not self.skipFirstSelfPlay or i > 1:
                if self.args.num_workers > 1:
                    self.trainExamplesHistory.append(self._self_play_parallel())
                else:
                    self.trainExamplesHistory.append(self._self_play_sequential())

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning("移除最早的训练样本")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self._train_and_evaluate(trainExamples, i)
            scheduler.step()

        self.writer.close()

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f'checkpoint_{iteration}.pth.tar.examples')
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'训练样本文件 "{examplesFile}" 未找到！')
        else:
            log.info("从文件加载训练样本...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('加载完成！')
            self.skipFirstSelfPlay = True


# ============================================================
# Self-play worker functions (module-level for ProcessPoolExecutor)
# ============================================================

_worker_game = None
_worker_args = None
_worker_predict_fn = None


def _init_self_play_worker(game_args, args_dict, state_dict):
    global _worker_game, _worker_args, _worker_predict_fn
    _worker_game = GomokuGame(*game_args)
    nnet = NNetWrapper(_worker_game)
    nnet.nnet.load_state_dict(state_dict)
    dummy = _worker_game.getInitBoard()
    nnet.predict(dummy)
    _worker_args = dotdict(args_dict)
    _worker_predict_fn = nnet.predict


def _run_self_play_episode(_unused):
    global _worker_game, _worker_args, _worker_predict_fn
    return Coach._run_episode(_worker_game, _worker_args, _worker_predict_fn)


_arena_game = None
_arena_args = None
_arena_nnet_new = None
_arena_nnet_old = None


def _init_arena_worker(game_args, new_state, old_state, args_dict):
    global _arena_game, _arena_args, _arena_nnet_new, _arena_nnet_old
    _arena_game = GomokuGame(*game_args)
    _arena_args = dotdict(args_dict)

    _arena_nnet_new = NNetWrapper(_arena_game)
    _arena_nnet_new.nnet.load_state_dict(new_state)
    _arena_nnet_old = NNetWrapper(_arena_game)
    _arena_nnet_old.nnet.load_state_dict(old_state)

    dummy = _arena_game.getInitBoard()
    _arena_nnet_new.predict(dummy)
    _arena_nnet_old.predict(dummy)


def _arena_play_game(new_first):
    global _arena_game, _arena_args, _arena_nnet_new, _arena_nnet_old

    mcts_new = MCTS(_arena_game, _arena_nnet_new, _arena_args)
    mcts_old = MCTS(_arena_game, _arena_nnet_old, _arena_args)

    if new_first:
        first_mcts, second_mcts = mcts_new, mcts_old
    else:
        first_mcts, second_mcts = mcts_old, mcts_new

    players = [second_mcts, None, first_mcts]
    curPlayer = 1
    board = _arena_game.getInitBoard()

    while _arena_game.getGameEnded(board, curPlayer) == 0:
        canonical = _arena_game.getCanonicalForm(board, curPlayer)
        action = int(np.argmax(players[curPlayer + 1].getActionProb(canonical, temp=0)))
        valids = _arena_game.getValidMoves(canonical, 1)
        if valids[action] == 0:
            action = int(np.random.choice(np.flatnonzero(valids)))
        board, curPlayer = _arena_game.getNextState(board, curPlayer, action)

    game_result = curPlayer * _arena_game.getGameEnded(board, curPlayer)

    if game_result > 0.5:
        return 1 if new_first else -1
    elif game_result < -0.5:
        return -1 if new_first else 1
    return 0


# ============================================================
# 训练入口
# ============================================================

DEFAULT_ARGS = dotdict({
    'numIters': 200,
    'numEps': 200,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 100000,
    'numMCTSSims': 800,
    'arenaCompare': 40,
    'cpuct': 1.0,
    'dirichlet_alpha': 1.0,
    'dirichlet_epsilon': 0.25,
    'num_workers': 8,
    'checkpoint': CHECKPOINT_DIR,
    'load_model': False,
    'load_folder_file': (CHECKPOINT_DIR, 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,
})


def setup_file_logging():
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
