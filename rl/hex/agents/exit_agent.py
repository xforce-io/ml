from __future__ import annotations
from hex.agents.agent import Agent, create_random_agent
from hex.config import ExperimentConfig
from hex.hex import Action, Board, State
from hex.agents.mcts_agent import MCTSNode, MCTSPolicy
from hex.experiment import HexGameExperiment
from hex.rl_basic import Episode, RandomPolicy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os

@dataclass
class ExitConfig:
    """Expert Iteration配置"""
    # MCTS相关配置
    simulations_per_move: int = 1600
    max_depth: int = 100
    c: float = 0.80
    base_rollouts_per_leaf: int = 40
    
    # 训练相关配置
    batch_size: int = 128
    memory_size: int = 100000
    num_iterations: int = 100
    self_play_games: int = 100
    temperature: float = 1.0
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_channels: int = 128
    policy_channels: int = 32
    value_hidden_size: int = 256
    name: str = "ExIt-Agent"
    
    # 网络相关配置
    use_network: bool = False  # 是否使用神经网络
    model_path: Optional[str] = None  # 模型路径
    model_dir: str = "data/models"  # 模型目录

class HexNet(nn.Module):
    """Hex游戏的神经网络模型"""
    def __init__(self, 
                 board_size: int, 
                 num_channels: int, 
                 policy_channels: int = 32,
                 value_hidden_size: int = 256):  # 添加value_hidden_size参数
        super().__init__()
        self.board_size = board_size
        
        # 共享特征提取层
        self.conv1 = nn.Conv2d(3, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        
        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, policy_channels, 1)
        self.policy_fc = nn.Linear(policy_channels * board_size * board_size, 
                                 board_size * board_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, policy_channels, 1)
        self.value_fc1 = nn.Linear(policy_channels * board_size * board_size, value_hidden_size)
        self.value_fc2 = nn.Linear(value_hidden_size, 1)

    def _preprocess_state(self, state: State) -> torch.Tensor:
        """将状态转换为神经网络输入格式"""
        # 创建3个通道：当前玩家棋子、对手棋子、空位
        current_player = state.current_player
        opponent = 3 - current_player
        
        tensor = torch.zeros(3, self.board_size, self.board_size)
        board = torch.tensor(state.board)
        
        tensor[0] = (board == current_player).float()
        tensor[1] = (board == opponent).float()
        tensor[2] = (board == 0).float()
        
        return tensor.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 策略头
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, self.policy_conv.out_channels * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # 价值头
        value = F.relu(self.value_conv(x))
        value = value.view(-1, self.value_conv.out_channels * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

# ... 其他代码保持不变 ... 