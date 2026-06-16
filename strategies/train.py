"""
策略训练入口脚本
训练 Q-Learning 策略

注意：Alpha-Beta、MCTS、启发式策略无需训练。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.q_learning import QLearningStrategy


def train_q_learning(episodes=1000):
    """训练Q-Learning策略"""
    print("=" * 50)
    print("训练 Q-Learning 策略")
    print("=" * 50)

    strategy = QLearningStrategy(
        board_size=15,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )

    strategy.train(episodes=episodes)
    return strategy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练五子棋策略")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="训练局数")

    args = parser.parse_args()
    train_q_learning(args.episodes)

    print("\nQ-Learning 训练完成！")
