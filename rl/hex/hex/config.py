from dataclasses import dataclass
import logging
import os
from typing import Optional

@dataclass
class ExperimentConfig:
    """实验全局配置"""
    # 基础配置
    num_cores: int  # 必需参数放在最前面
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

    def __post_init__(self):
        """验证配置参数"""
        assert self.total_rounds % self.statistics_rounds == 0, "total_rounds必须能被statistics_rounds整除"
        assert self.statistics_rounds % self.num_cores == 0, "statistics_rounds必须能被num_cores整除"
        
        # 创建必要的目录
        for directory in [self.data_dir, self.model_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)

@dataclass
class MCTSConfig:
    """MCTS算法配置"""
    simulations: int = 1600
    max_depth: int = 100
    c: float = 0.80
    use_rave: bool = False
    rave_constant: float = 300
    selection_strategy: str = 'robust'
    base_rollouts_per_leaf: int = 40
    name: str = "MCTS-Advanced"

@dataclass
class ExitConfig:
    """Expert Iteration配置"""
    mcts_config: MCTSConfig = MCTSConfig()
    
    # 训练相关配置
    batch_size: int = 128
    memory_size: int = 100000
    num_iterations: int = 200
    self_play_games: int = 200
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

