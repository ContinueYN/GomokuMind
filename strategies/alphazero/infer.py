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
logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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
        canonical = go_board_to_canonical(board, player)
        self.mcts.reset()
        pi = self.mcts.getActionProb(canonical, temp=0)
        action = int(np.argmax(pi))
        row, col = divmod(action, BOARD_SIZE)
        return row, col


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python infer.py <model_path>"}))
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"Model not found: {model_path}"}))
        sys.exit(1)

    try:
        engine = InferenceEngine(model_path)
    except Exception as e:
        log.exception("Failed to initialize inference engine")
        print(json.dumps({"error": f"Model load failed: {e}"}), flush=True)
        sys.exit(1)

    log.info("Inference engine ready, waiting for requests...")
    print(json.dumps({"status": "ready"}), flush=True)

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
