"""五子棋基础规则函数，供 alphazero infer.py 和 alphazero_train.py 使用。"""

from __future__ import annotations

import numpy as np


def check_winner_from(board: np.ndarray, player: int, r: int, c: int,
                      connect: int = 5) -> bool:
    """检查以 (r,c) 为落子点是否连成 connect 子。只读，O(4*connect)。"""
    rows, cols = board.shape

    count = 1
    for cc in range(c - 1, -1, -1):
        if board[r, cc] != player: break
        count += 1
    for cc in range(c + 1, cols):
        if board[r, cc] != player: break
        count += 1
    if count >= connect: return True

    count = 1
    for rr in range(r - 1, -1, -1):
        if board[rr, c] != player: break
        count += 1
    for rr in range(r + 1, rows):
        if board[rr, c] != player: break
        count += 1
    if count >= connect: return True

    count = 1
    for i in range(1, min(r, c) + 1):
        if board[r - i, c - i] != player: break
        count += 1
    for i in range(1, min(rows - r, cols - c)):
        if board[r + i, c + i] != player: break
        count += 1
    if count >= connect: return True

    count = 1
    for i in range(1, min(r, cols - 1 - c) + 1):
        if board[r - i, c + i] != player: break
        count += 1
    for i in range(1, min(rows - r, c + 1)):
        if board[r + i, c - i] != player: break
        count += 1
    if count >= connect: return True

    return False
