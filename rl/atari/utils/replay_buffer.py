import random
import time
import numpy as np
from collections import deque, namedtuple
import torch

# 定义一个具名元组来表示单个经验
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """基于张量的高效经验回放缓冲区 (数据存储在指定设备)"""
    def __init__(self, capacity, device='mps'):
        """
        Args:
            capacity (int): 缓冲区的最大容量
            device (str): 存储张量的设备 ('cpu', 'cuda', 或 'mps')
        """
        self.capacity = capacity
        self.device = torch.device(device) # 确保 device 是 torch.device 对象
        self.current_size = 0
        self.ptr = 0  # 指向下一个要插入的位置

        # 缓冲区张量将在 _initialize_buffers 中在目标设备上分配
        self.states = None
        self.next_states = None
        # 在目标设备上预分配固定大小的张量
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

    def _initialize_buffers(self, state):
        """根据第一个状态在目标设备上初始化缓冲区"""
        # 确保 state 是 NumPy 数组以获取形状
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy() # 假设 state 可能来自环境 (CPU)
        elif not isinstance(state, np.ndarray):
            state_np = np.array(state)
        else:
            state_np = state
            
        state_shape = state_np.shape
        # 在目标设备上分配状态缓冲区
        self.states = torch.zeros((self.capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.capacity, *state_shape), dtype=torch.float32, device=self.device)
        print(f"ReplayBuffer 在设备 {self.device} 上初始化完成，状态形状: {state_shape}")

    def push(self, state, action, reward, next_state, done):
        """保存一个经验 transition，数据直接存到目标设备"""
        # 如果是第一次添加数据，初始化缓冲区
        if self.states is None:
            # 注意：这里假设传入的第一个 state 来自 CPU 或 NumPy
            self._initialize_buffers(state)

        # 将输入数据转换为张量并移动到目标设备
        # 注意: 假设输入 state, next_state 是 numpy 数组或 list, action/reward/done 是标量或 numpy 数组
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action_tensor = torch.as_tensor(action, dtype=torch.long).to(self.device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32).to(self.device)

        # 存储到目标设备上的缓冲区
        self.states[self.ptr] = state_tensor
        self.actions[self.ptr] = action_tensor
        self.rewards[self.ptr] = reward_tensor
        self.next_states[self.ptr] = next_state_tensor
        self.dones[self.ptr] = done_tensor

        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size):
        """从缓冲区中随机采样一个批次的经验 (数据已在目标设备)"""
        if self.current_size < batch_size:
            raise ValueError(f"缓冲区经验数量不足 ({self.current_size} < {batch_size})")

        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回当前缓冲区中的经验数量"""
        return self.current_size 