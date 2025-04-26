from abc import ABC, abstractmethod
import torch
import os

class Algo(ABC):
    """强化学习算法的抽象基类"""
    def __init__(self, env, config, device):
        """
        Args:
            env: 绑定的 Gymnasium 环境实例 (已包装)
            config: 包含该算法特定配置的对象 (例如 DqnConfig)
            device: PyTorch 设备 (torch.device)
        """
        self.env = env
        self.config = config
        self.device = device
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.steps_done = 0 # 跟踪总交互步数，由外部 Experiment 更新

    @abstractmethod
    def selectAction(self, state, deterministic=False):
        """根据当前状态选择动作

        Args:
            state (np.ndarray): 当前环境观察值 (C, H, W)
            deterministic (bool): 是否使用确定性策略 (例如，评估时关闭探索)

        Returns:
            int: 选择的动作索引
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """处理一个时间步的经验，并可能执行学习更新

        Args:
            state (np.ndarray): 当前状态
            action (int): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            done (bool): Episode 是否结束

        Returns:
            float or None: 如果执行了学习，返回损失值；否则返回 None
        """
        pass

    @abstractmethod
    def save(self, directory, filename):
        """保存模型/算法状态"""
        # 确保目录存在
        if not os.path.exists(directory):
            os.makedirs(directory)
        pass

    @abstractmethod
    def load(self, filepath):
        """加载模型/算法状态

        Returns:
            bool: 加载成功返回 True，否则返回 False
        """
        pass
        
    def updateStepsDone(self, steps):
        """由外部（如 Experiment）更新当前步数
        
        Args:
            steps (int): 当前步数
        """
        self.steps_done = steps 