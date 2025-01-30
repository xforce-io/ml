from dataclasses import dataclass
import logging
import os
from typing import Optional

import torch

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
    simulations: int = 100
    max_depth: int = 50
    c: float = 0.80
    use_rave: bool = False
    rave_constant: float = 300
    selection_strategy: str = 'robust'
    base_rollouts_per_leaf: int = 20
    name: str = "MCTS-Advanced"

@dataclass
class ExitConfig:
    """Expert Iteration配置"""
    def __init__(self):
        # 现有配置
        self.num_iterations = 1
        self.self_play_games = 20
        self.batch_size = 32
        self.memory_size = 10000
        self.save_interval = 10
        self.model_dir = "data/models"
        self.model_path = None
        self.use_network = True
        self.temperature = 1.0
        self.num_channels = 128
        self.policy_channels = 32
        self.learning_rate = 0.001
        self.weight_decay = 0.001
        self.mcts_config = MCTSConfig()
        self.parallel_self_play = True
        self.parallel_eval = True 
        
        # 添加网络服务器配置
        self.network_server_host = "127.0.0.1"
        self.network_server_port = 8123

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
