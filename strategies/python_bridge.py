"""
Python策略桥接 - 通过stdin/stdout与Go后端通信

输入格式 (stdin JSON):
  {"strategy": "q-learning", "board": [[...], ...], "player": 1 | -1}

输出格式 (stdout JSON):
  {"row": 7, "col": 7}
"""

import sys
import os
import json
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOARD_SIZE = 15


def get_nearby_moves(board: np.ndarray, player: int):
    """获取已有棋子周围2格内的空位"""
    seen = set()
    moves = []

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != 0:
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        r, c = i + di, j + dj
                        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == 0:
                            key = (r, c)
                            if key not in seen:
                                seen.add(key)
                                moves.append((r, c))

    if not moves:
        # 空棋盘，下天元
        moves.append((7, 7))
    return moves


def check_win(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """检查落子后是否获胜"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for i in range(1, 5):
            r, c = row + dr * i, col + dc * i
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
            else:
                break
        for i in range(1, 5):
            r, c = row - dr * i, col - dc * i
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
    return False


def find_winning_move(board: np.ndarray, player: int):
    """检测是否有直接获胜的落子"""
    candidates = get_nearby_moves(board, player)
    for r, c in candidates:
        test = board.copy()
        test[r][c] = player
        if check_win(test, r, c, player):
            return (r, c)
    return None


def heuristic_fallback(board: np.ndarray, player: int):
    """当Python策略模型不可用时的启发式兜底"""
    # 检查直接获胜
    win = find_winning_move(board, player)
    if win:
        return win

    # 检查需要堵对手的
    opp = -player
    block = find_winning_move(board, opp)
    if block:
        return block

    # 评分：评估每个候选位置
    candidates = get_nearby_moves(board, player)
    best_score = -999999
    best_move = (7, 7)

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for r, c in candidates:
        score = 0
        for dr, dc in directions:
            count = 1
            open_ends = 0
            # 正方向
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == player:
                        count += 1
                    elif board[nr][nc] == 0:
                        open_ends += 1
                        break
                    else:
                        break
                else:
                    break
            # 反方向
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == player:
                        count += 1
                    elif board[nr][nc] == 0:
                        open_ends += 1
                        break
                    else:
                        break
                else:
                    break

            # 连子评分
            if count >= 5:
                score += 1000000
            elif count == 4 and open_ends == 2:
                score += 100000
            elif count == 4 and open_ends == 1:
                score += 10000
            elif count == 3 and open_ends == 2:
                score += 5000
            elif count == 3 and open_ends == 1:
                score += 500
            elif count == 2 and open_ends == 2:
                score += 200
            elif count == 2 and open_ends == 1:
                score += 50

        # 防守得分
        for dr, dc in directions:
            count = 1
            open_ends = 0
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == opp:
                        count += 1
                    elif board[nr][nc] == 0:
                        open_ends += 1
                        break
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == opp:
                        count += 1
                    elif board[nr][nc] == 0:
                        open_ends += 1
                        break
                    else:
                        break
                else:
                    break

            if count >= 4:
                score += 8000  # 堵对手的四连很重要

        if score > best_score:
            best_score = score
            best_move = (r, c)

    return best_move


def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse input: {e}"}))
        return

    strategy_name = data.get("strategy", "")
    board_data = data.get("board", [])
    player = data.get("player", 1)

    # 转换为numpy数组
    board = np.array(board_data, dtype=int)

    if strategy_name == "q-learning":
        move = handle_q_learning(board, player)
    else:
        print(json.dumps({"error": f"Unknown strategy: {strategy_name}"}))
        return

    print(json.dumps({"row": move[0], "col": move[1]}))


def handle_q_learning(board: np.ndarray, player: int):
    """调用Q-Learning策略"""
    try:
        from strategies.q_learning import QLearningStrategy
        strategy = QLearningStrategy()
        strategy.load_model()
        move = strategy.get_move(board, player)
        return move
    except Exception as e:
        return heuristic_fallback(board, player)


if __name__ == "__main__":
    main()
