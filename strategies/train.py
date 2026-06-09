"""
策略训练入口脚本
训练Q-Learning和PPO策略
"""

import sys
import os

# 添加策略路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'q-learning'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ppo'))

from q_learning import QLearningStrategy
from ppo_strategy import PPOStrategy

def train_q_learning(episodes=1000):
    """训练Q-Learning策略"""
    print("=" * 50)
    print("训练Q-Learning策略")
    print("=" * 50)

    strategy = QLearningStrategy(
        board_size=15,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )

    strategy.train(episodes=episodes)
    return strategy

def train_ppo(episodes=500):
    """训练PPO策略"""
    print("=" * 50)
    print("训练PPO (Actor-Critic) 策略")
    print("=" * 50)

    strategy = PPOStrategy(
        board_size=15,
        lr=0.001,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4
    )

    strategy.train(episodes=episodes)
    return strategy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练五子棋策略")
    parser.add_argument("--strategy", choices=["q-learning", "ppo", "all"], default="all",
                        help="选择要训练的策略")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="训练局数")

    args = parser.parse_args()

    if args.strategy == "q-learning" or args.strategy == "all":
        train_q_learning(args.episodes)

    if args.strategy == "ppo" or args.strategy == "all":
        train_ppo(args.episodes // 2)  # PPO训练较慢，使用一半局数

    print("\n所有策略训练完成！")
