"""
Q-Learning策略 - 经典强化学习
使用Q表学习，通过状态抽象处理15x15棋盘的大状态空间
"""

import numpy as np
import pickle
import os
from typing import Tuple, List

class QLearningStrategy:
    def __init__(self, board_size=15, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.board_size = board_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.model_path = os.path.join(os.path.dirname(__file__), "q_table.pkl")

    def name(self) -> str:
        return "Q-Learning"

    def _state_to_key(self, board: np.ndarray) -> str:
        """将棋盘状态抽象为Q表键值
        使用局部特征抽象：只关注棋子分布模式，而非绝对位置
        """
        # 简化：将棋盘划分为3x3的区域，每个区域统计黑/白/空的数量
        region_size = self.board_size // 3
        features = []

        for i in range(3):
            for j in range(3):
                r_start, r_end = i * region_size, (i + 1) * region_size
                c_start, c_end = j * region_size, (j + 1) * region_size
                region = board[r_start:r_end, c_start:c_end]

                black_count = np.sum(region == 1)
                white_count = np.sum(region == -1)
                empty_count = np.sum(region == 0)

                features.extend([black_count, white_count, empty_count])

        # 加上当前玩家
        return str(features)

    def get_valid_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """获取合法落子位置"""
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def get_move(self, board: np.ndarray, player: int) -> Tuple[int, int]:
        """根据epsilon-greedy策略选择落子"""
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return (7, 7)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return valid_moves[np.random.randint(len(valid_moves))]

        state_key = self._state_to_key(board)
        if state_key not in self.q_table:
            return valid_moves[np.random.randint(len(valid_moves))]

        q_values = self.q_table[state_key]
        best_q = -np.inf
        best_moves = []

        for move in valid_moves:
            move_idx = move[0] * self.board_size + move[1]
            if move_idx < len(q_values):
                q_val = q_values[move_idx]
                if q_val > best_q:
                    best_q = q_val
                    best_moves = [move]
                elif q_val == best_q:
                    best_moves.append(move)

        if best_moves:
            return best_moves[np.random.randint(len(best_moves))]

        return valid_moves[np.random.randint(len(valid_moves))]

    def update(self, state: np.ndarray, action: Tuple[int, int], reward: float, next_state: np.ndarray, done: bool):
        """更新Q表"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        action_idx = action[0] * self.board_size + action[1]

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.board_size * self.board_size)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.board_size * self.board_size)

        current_q = self.q_table[state_key][action_idx]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_key])

        # Q-Learning更新公式
        self.q_table[state_key][action_idx] = current_q + self.lr * (target_q - current_q)

    def train(self, episodes: int = 1000):
        """自我对弈训练"""
        print(f"开始Q-Learning训练，共{episodes}局...")

        for episode in range(episodes):
            board = np.zeros((self.board_size, self.board_size), dtype=int)
            current_player = 1
            done = False
            last_state = None
            last_action = None

            while not done:
                state = board.copy()
                state_key = self._state_to_key(state)

                if state_key not in self.q_table:
                    self.q_table[state_key] = np.zeros(self.board_size * self.board_size)

                # 选择动作
                valid_moves = self.get_valid_moves(board)
                if not valid_moves:
                    break

                if np.random.random() < self.epsilon:
                    action = valid_moves[np.random.randint(len(valid_moves))]
                else:
                    q_values = self.q_table[state_key]
                    best_q = -np.inf
                    best_moves = []
                    for move in valid_moves:
                        idx = move[0] * self.board_size + move[1]
                        if idx < len(q_values):
                            if q_values[idx] > best_q:
                                best_q = q_values[idx]
                                best_moves = [move]
                            elif q_values[idx] == best_q:
                                best_moves.append(move)
                    action = best_moves[np.random.randint(len(best_moves))] if best_moves else valid_moves[0]

                # 执行动作
                board[action[0]][action[1]] = current_player

                # 检查胜负
                reward = 0
                if self.check_win(board, action[0], action[1], current_player):
                    reward = 1 if current_player == 1 else -1
                    done = True
                elif len(self.get_valid_moves(board)) == 0:
                    reward = 0
                    done = True

                # 更新Q表
                if last_state is not None:
                    self.update(last_state, last_action, -reward if not done else reward, board, done)

                last_state = state
                last_action = action
                current_player = -current_player

            if (episode + 1) % 100 == 0:
                print(f"完成{episode + 1}局训练")

        # 保存模型
        self.save_model()
        print("训练完成！")

    def check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """检查是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r][c] == player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r][c] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def save_model(self):
        """保存Q表"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q表已保存到{self.model_path}")

    def load_model(self):
        """加载Q表"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q表已从{self.model_path}加载")
        else:
            print("未找到预训练的Q表，使用空表")
