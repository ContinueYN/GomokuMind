"""
PPO (Proximal Policy Optimization) 策略 - Actor-Critic架构
2025-2026主流强化学习算法，使用自对弈训练
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from typing import Tuple, List

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GomokuNetwork(nn.Module):
    """Actor-Critic网络
    Actor: 输出每个位置的落子概率
    Critic: 输出当前状态的价值评估
    """
    def __init__(self, board_size=15):
        super().__init__()
        self.board_size = board_size

        # 卷积层提取空间特征
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3通道：黑/白/空
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Actor头 - 策略输出
        self.actor_fc = nn.Linear(128 * board_size * board_size, board_size * board_size)

        # Critic头 - 价值评估
        self.critic_fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.critic_fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, 3, board_size, board_size)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x_flat = x.view(x.size(0), -1)

        # Actor输出
        action_probs = torch.softmax(self.actor_fc(x_flat), dim=-1)

        # Critic输出
        value = torch.tanh(self.critic_fc2(self.relu(self.critic_fc1(x_flat))))

        return action_probs, value

class PPOStrategy:
    def __init__(self, board_size=15, lr=0.001, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.board_size = board_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = GomokuNetwork(board_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.model_path = os.path.join(os.path.dirname(__file__), "ppo_model.pth")

    def name(self) -> str:
        return "PPO (Actor-Critic)"

    def board_to_tensor(self, board: np.ndarray) -> torch.Tensor:
        """将棋盘转换为网络输入
        3个通道：黑棋/白棋/空位
        """
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (board == 1).astype(np.float32)  # 黑棋
        state[1] = (board == -1).astype(np.float32)  # 白棋
        state[2] = (board == 0).astype(np.float32)  # 空位
        return torch.FloatTensor(state).unsqueeze(0).to(device)

    def get_valid_moves_mask(self, board: np.ndarray) -> np.ndarray:
        """获取合法落子掩码"""
        return (board == 0).flatten()

    def get_move(self, board: np.ndarray, player: int) -> Tuple[int, int]:
        """获取推荐落子"""
        state = self.board_to_tensor(board)
        with torch.no_grad():
            action_probs, _ = self.policy(state)

        # 应用合法落子掩码
        mask = self.get_valid_moves_mask(board)
        action_probs = action_probs.squeeze(0).cpu().numpy()
        action_probs = action_probs * mask
        action_probs = action_probs / (action_probs.sum() + 1e-10)

        # 选择概率最高的合法位置
        action_idx = np.argmax(action_probs)
        return (action_idx // self.board_size, action_idx % self.board_size)

    def get_move_with_exploration(self, board: np.ndarray) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """带探索的动作选择（训练用）"""
        state = self.board_to_tensor(board)
        action_probs, value = self.policy(state)

        # 应用合法落子掩码
        mask = self.get_valid_moves_mask(board)
        masked_probs = action_probs.squeeze(0) * torch.FloatTensor(mask).to(device)
        masked_probs = masked_probs / (masked_probs.sum() + 1e-10)

        dist = Categorical(masked_probs)
        action = dist.sample()

        return (
            action.item() // self.board_size,
            action.item() % self.board_size,
            dist.log_prob(action),
            value.squeeze()
        )

    def train(self, episodes: int = 500):
        """PPO自对弈训练"""
        print(f"开始PPO训练，共{episodes}局...")

        for episode in range(episodes):
            # 收集轨迹
            states, actions, rewards, log_probs, values = [], [], [], [], []
            board = np.zeros((self.board_size, self.board_size), dtype=int)
            current_player = 1
            done = False

            while not done:
                state = board.copy()
                action_r, action_c, log_prob, value = self.get_move_with_exploration(board)

                # 执行动作
                board[action_r][action_c] = current_player

                # 检查胜负
                reward = 0
                if self.check_win(board, action_r, action_c, current_player):
                    reward = 1 if current_player == 1 else -1
                    done = True
                elif np.sum(board == 0) == 0:
                    reward = 0
                    done = True

                states.append(state)
                actions.append((action_r, action_c))
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

                current_player = -current_player

            # PPO更新
            self._ppo_update(states, actions, rewards, log_probs, values)

            if (episode + 1) % 50 == 0:
                print(f"完成{episode + 1}局训练")

        self.save_model()
        print("PPO训练完成！")

    def _ppo_update(self, states, actions, rewards, old_log_probs, old_values):
        """PPO策略更新"""
        # 计算优势函数
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(device)
        old_values = torch.stack(old_values).detach()
        advantages = returns - old_values

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多次更新
        for _ in range(self.k_epochs):
            # 重新计算当前策略下的log_probs和values
            new_log_probs = []
            new_values = []

            for state, action in zip(states, actions):
                state_tensor = self.board_to_tensor(state)
                action_probs, value = self.policy(state_tensor)

                mask = self.get_valid_moves_mask(state)
                masked_probs = action_probs.squeeze(0) * torch.FloatTensor(mask).to(device)
                masked_probs = masked_probs / (masked_probs.sum() + 1e-10)

                dist = Categorical(masked_probs)
                action_idx = action[0] * self.board_size + action[1]
                new_log_probs.append(dist.log_prob(torch.tensor(action_idx).to(device)))
                new_values.append(value.squeeze())

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)

            # PPO损失
            ratios = torch.exp(new_log_probs - torch.stack(old_log_probs).detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values, returns)
            loss = actor_loss + 0.5 * critic_loss

            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
        """保存模型"""
        torch.save(self.policy.state_dict(), self.model_path)
        print(f"模型已保存到{self.model_path}")

    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path):
            self.policy.load_state_dict(torch.load(self.model_path, map_location=device))
            self.policy.eval()
            print(f"模型已从{self.model_path}加载")
        else:
            print("未找到预训练模型，使用随机初始化")
