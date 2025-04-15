import random
import numpy as np
from collections import deque, namedtuple
import torch

# 定义一个具名元组来表示单个经验
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """基于张量的高效经验回放缓冲区"""
    def __init__(self, capacity, device='mps'):
        """
        Args:
            capacity (int): 缓冲区的最大容量
            device (str): 存储张量的设备 ('cpu' 或 'mps')
        """
        self.capacity = capacity
        self.device = device
        self.current_size = 0
        self.ptr = 0  # 指向下一个要插入的位置
        
        # 在 CPU 上预分配固定大小的张量
        self.states = None
        self.next_states = None
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)

    def _initialize_buffers(self, state):
        """根据第一个状态初始化缓冲区"""
        state_shape = state.shape if isinstance(state, np.ndarray) else np.array(state).shape
        self.states = torch.zeros((self.capacity, *state_shape), dtype=torch.float32)
        self.next_states = torch.zeros((self.capacity, *state_shape), dtype=torch.float32)

    def push(self, state, action, reward, next_state, done):
        """保存一个经验 transition"""
        # 如果是第一次添加数据，初始化缓冲区
        if self.states is None:
            self._initialize_buffers(state)
            
        # 转换为张量并存储在 CPU 上
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.long)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        
        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size):
        """从缓冲区中随机采样一个批次的经验"""
        if self.current_size < batch_size:
            raise ValueError(f"缓冲区经验数量不足 ({self.current_size} < {batch_size})")
        
        # 在 CPU 上生成随机索引
        indices = torch.randint(0, self.current_size, (batch_size,))
        
        # 在 CPU 上采样数据
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        
        # 将数据转移到目标设备
        return (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device)
        )
    
    def __len__(self):
        """返回当前缓冲区中的经验数量"""
        return self.current_size 