from dataclasses import dataclass, field, fields
import logging
import os
from typing import Optional

import torch

@dataclass
class ExperimentConfig:
    """实验全局配置"""
    # 基础配置
    num_cores: int  # 必需参数放在最前面
    num_games_to_evaluate: int = 50
    board_size: int = 5
    total_rounds: int = 400
    statistics_rounds: int = 200
    timeout: int = 600  # 批次超时时间（秒）
    
    # 路径配置
    data_dir: str = "data"
    model_dir: str = "data/models"
    log_dir: str = "data/logs"
    
    # 日志配置
    log_level: int = logging.INFO
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s'

    def json(self) -> dict:
        """返回配置的JSON表示"""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

    def __post_init__(self):
        """验证配置参数"""
        assert self.total_rounds % self.statistics_rounds == 0, "total_rounds必须能被statistics_rounds整除"
        assert self.statistics_rounds % self.num_cores == 0, "statistics_rounds必须能被num_cores整除"
        
        # 创建必要的目录
        for directory in [self.data_dir, self.model_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)

@dataclass
class MCTSConfig:
    """MCTS（蒙特卡洛树搜索）算法配置"""
    # 搜索参数
    simulations: int = 200          # 每次决策的模拟次数
    max_depth: int = 60            # 最大搜索深度
    c: float = 0.80               # UCB公式中的探索常数
    
    # RAVE参数
    use_rave: bool = False         # 是否使用RAVE
    rave_constant: float = 300     # RAVE常数
    
    # 策略参数
    selection_strategy: str = 'robust'  # 节点选择策略：'robust' 或 'max'
    base_rollouts_per_leaf: int = 20   # 每个叶节点的基础rollout次数
    
    # 其他
    name: str = "MCTS-Advanced"    # 算法名称

    def json(self) -> dict:
        """返回配置的JSON表示"""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

@dataclass
class ExitConfig:
    """Expert Iteration（专家迭代）配置"""
    # 训练参数
    num_steps_per_epoch: int = 20    # 每个epoch的步数
    num_epochs: int = 10            # 训练的epoch数
    batch_size: int = 32            # 批次大小
    memory_size: int = 10000        # 经验回放缓冲区大小
    
    # 自对弈参数
    parallel_self_play: bool = True # 是否并行进行自对弈
    parallel_eval: bool = True      # 是否并行进行评估
    
    # 模型参数
    num_channels: int = 128         # 卷积层通道数
    policy_channels: int = 32       # 策略头通道数
    value_channels: int = 32        # 价值头通道数
    use_network: bool = True        # 是否使用神经网络
    
    # 优化器参数
    learning_rate: float = 0.01     # 学习率
    weight_decay: float = 0.001     # 权重衰减
    warmup_steps: int = 1000        # 预热步数
    
    # 保存和加载
    save_interval: int = 10         # 模型保存间隔
    model_dir: str = "data/models"  # 模型保存目录
    model_path: Optional[str] = None # 预训练模型路径
    
    # 网络服务配置
    network_server_host: str = "127.0.0.1"
    network_server_port: int = 8123
    
    # MCTS配置
    mcts_config: MCTSConfig = field(default_factory=MCTSConfig)

    def __post_init__(self):
        """初始化派生属性"""
        self.num_steps = self.num_steps_per_epoch * self.num_epochs

    def json(self) -> dict:
        """返回配置的JSON表示"""
        result = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != 'mcts_config'
        }
        result['mcts_config'] = self.mcts_config.json()
        return result

@dataclass
class DynaQConfig:
    """DynaQ算法配置"""
    algorithm_type: str = 'DynaQ'
    initial_learning_rate: float = 0.2
    final_learning_rate: float = 0.01
    initial_epsilon: float = 0.3
    final_epsilon: float = 0.05
    gamma: float = 0.99
    planning_steps: int = 200
    batch_size: int = 64
    memory_size: int = 50000
    name: str = "DynaQ"

def get_current_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
