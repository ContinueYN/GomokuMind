#!/usr/bin/env python3
"""
AlphaZero 持久推理服务 — 由 Go 端通过 stdin/stdout 管道调用。

启动方式（由 Go 端管理进程）：
    python infer.py <model_path>

协议（JSON Lines，每行一个请求）：
    输入:  {"board": [[0,1,2,...], ...], "player": 1}
    输出:  {"row": 7, "col": 7}

    board: 15x15, 0=空 1=黑 2=白
    player: 1=黑棋回合, 2=白棋回合
"""

import sys
import os
import json
import logging
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import torch

from alphazero_train import GomokuGame, NNetWrapper, MCTS, dotdict
from gomoku import check_winner_from

BOARD_SIZE = 15
CONNECT = 5

log = logging.getLogger("alphazero.infer")
# 日志输出到 stderr，避免污染 stdout 上的 JSON Lines 协议
logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# 推理场景配置：模拟次数减半（400 vs 训练时 800）以降低延迟
# cpuct=1.0 为标准 AlphaZero 设定
MCTS_SIMS = 400
CPUCT = 1.0


def go_board_to_canonical(board: np.ndarray, player: int) -> np.ndarray:
    """将 Go 格式棋盘 (0=空, 1=黑, 2=白) 转为 AlphaZero 规范形式。

    AlphaZero 规范: 当前玩家棋子 → 1, 对手棋子 → -1.
    player: 1=黑棋, 2=白棋
    """
    b = np.asarray(board, dtype=np.int8)
    mult = 1 if player == 1 else -1
    az = np.where(b == 0, 0, np.where(b == 1, mult, -mult)).astype(np.int8)
    return az


class InferenceEngine:
    """AlphaZero 推理引擎 — 加载模型并处理每次落子请求。

    每次 predict 调用是独立的：
    1. 将 Go 棋盘格式转为 AlphaZero 规范形式
    2. 重置 MCTS 搜索树（新局面的搜索空间完全独立）
    3. 执行 400 次 MCTS 模拟
    4. 返回最优落子坐标
    """

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s", self.device)

        self.game = GomokuGame(rows=BOARD_SIZE, cols=BOARD_SIZE, connect=CONNECT)
        self.nnet = NNetWrapper(self.game, device=self.device)
        self.nnet.load_checkpoint(filename=model_path)
        log.info("Model loaded from %s", model_path)

        self.args = dotdict({"numMCTSSims": MCTS_SIMS, "cpuct": CPUCT})
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def predict(self, board: np.ndarray, player: int) -> tuple:
        """给定棋盘和当前玩家，返回最优落子 (row, col)。

        player ∈ {1, 2}，其中 1=黑棋回合, 2=白棋回合。
        """
        canonical = go_board_to_canonical(board, player)
        self.mcts.reset()  # 每步清空搜索树，避免旧状态污染
        pi = self.mcts.getActionProb(canonical, temp=0)  # 温度=0：确定性最强走法
        action = int(np.argmax(pi))
        row, col = divmod(action, BOARD_SIZE)
        return row, col


def main():
    """持久推理服务主循环。

    生命周期：
    1. 加载模型 → 打印 {"status": "ready"} 通知 Go 端
    2. 循环从 stdin 读取 JSON Lines 请求
    3. 每次请求执行 MCTS 搜索，输出 {"row": N, "col": M}
    4. 收到 "quit" 或 stdin 关闭时退出

    错误处理：所有异常都转为 JSON 错误响应输出，不导致进程崩溃。
    Go 端通过检查响应中是否有 "error" 字段判断成功与否。
    """
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python infer.py <model_path>"}))
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"Model not found: {model_path}"}))
        sys.exit(1)

    # 初始化引擎（包含模型加载 + 网络预热）
    try:
        engine = InferenceEngine(model_path)
    except Exception as e:
        log.exception("Failed to initialize inference engine")
        print(json.dumps({"error": f"Model load failed: {e}"}), flush=True)
        sys.exit(1)

    # 发送就绪信号 — Go 端阻塞等待此信号（60s 超时）
    log.info("Inference engine ready, waiting for requests...")
    print(json.dumps({"status": "ready"}), flush=True)

    # JSON Lines 事件循环
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
            continue

        board_data = req.get("board")
        player = req.get("player", 1)

        if board_data is None:
            print(json.dumps({"error": "Missing 'board' field"}), flush=True)
            continue

        try:
            row, col = engine.predict(np.array(board_data, dtype=int), player)
            print(json.dumps({"row": row, "col": col}), flush=True)
        except Exception as e:
            log.exception("Prediction failed")
            print(json.dumps({"error": str(e)}), flush=True)

    log.info("Inference server shutting down.")


if __name__ == "__main__":
    main()
