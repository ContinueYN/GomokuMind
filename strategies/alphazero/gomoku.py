"""五子棋基础规则函数，供 alphazero / minimax 策略使用。

本模块独立于训练/推理逻辑，仅提供纯规则判定和棋盘常量。
"""

from __future__ import annotations

import numpy as np


class GomokuEnv:
    """五子棋环境常量 — 与 game-engine 保持一致的 15×15 自由落子规则。"""
    ROWS = 15
    COLS = 15
    CONNECT = 5
    PLAYER_1 = 1  # 黑棋（先手）
    # PLAYER_2 = 2  # 白棋（后手），minimax 用 3-player 公式推导


def check_winner_from(board: np.ndarray, player: int, r: int, c: int,
                      connect: int = 5) -> bool:
    """检查以 (r,c) 为落子点是否连成 connect 子。

    算法：从落子点向四个方向（水平、垂直、两条对角线）各两端延伸，
    计数同色连续棋子数，任一方向达到 connect 即获胜。

    复杂度：O(4*connect)，只读操作，不修改棋盘。
    设计为以落子点为中心的增量判定，比全盘扫描高效得多。

    Args:
        board: shape=(rows, cols) 的棋盘数组
        player: 要检查的玩家（棋盘值等于 player 的位置）
        r, c: 最近落子坐标
        connect: 连子数阈值，默认 5

    Returns:
        True 如果 player 在 (r,c) 处形成 connect 连子
    """
    rows, cols = board.shape

    # ── 水平方向 (→) ──
    count = 1
    for cc in range(c - 1, -1, -1):        # 向左延伸
        if board[r, cc] != player: break
        count += 1
    for cc in range(c + 1, cols):          # 向右延伸
        if board[r, cc] != player: break
        count += 1
    if count >= connect: return True

    # ── 垂直方向 (↓) ──
    count = 1
    for rr in range(r - 1, -1, -1):        # 向上延伸
        if board[rr, c] != player: break
        count += 1
    for rr in range(r + 1, rows):          # 向下延伸
        if board[rr, c] != player: break
        count += 1
    if count >= connect: return True

    # ── 主对角线方向 (↘) ──
    count = 1
    for i in range(1, min(r, c) + 1):      # 左上延伸
        if board[r - i, c - i] != player: break
        count += 1
    for i in range(1, min(rows - r, cols - c)):  # 右下延伸
        if board[r + i, c + i] != player: break
        count += 1
    if count >= connect: return True

    # ── 反对角线方向 (↙) ──
    count = 1
    for i in range(1, min(r, cols - 1 - c) + 1):  # 右上延伸
        if board[r - i, c + i] != player: break
        count += 1
    for i in range(1, min(rows - r, c + 1)):      # 左下延伸
        if board[r + i, c - i] != player: break
        count += 1
    if count >= connect: return True

    return False
