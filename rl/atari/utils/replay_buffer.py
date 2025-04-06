import random
import numpy as np
from collections import deque, namedtuple
import torch

# 定义一个具名元组来表示单个经验
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """简单的经验回放缓冲区"""
    def __init__(self, capacity, device='cpu'):
        """
        Args:
            capacity (int): 缓冲区的最大容量
            device (str): 存储张量的设备 ('cpu' 或 'mps')
        """
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """保存一个经验 transition"""
        state_tensor = torch.from_numpy(state).float().to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state_tensor = torch.from_numpy(next_state).float().to(self.device)
        done_tensor = torch.tensor(done, dtype=torch.float).to(self.device)
        
        self.memory.append(Transition(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))

    def sample(self, batch_size):
        """从缓冲区中随机采样一个批次的经验"""
        if len(self.memory) < batch_size:
            raise ValueError(f"缓冲区经验数量不足 ({len(self.memory)} < {batch_size})")
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([t.state for t in batch])
        actions = torch.stack([t.action for t in batch])
        rewards = torch.stack([t.reward for t in batch])
        next_states = torch.stack([t.next_state for t in batch])
        dones = torch.stack([t.done for t in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回当前缓冲区中的经验数量"""
        return len(self.memory) 