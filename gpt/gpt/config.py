from dataclasses import dataclass
from typing import Optional
import yaml
import os

@dataclass
class GlobalConfig:
    cache_dir: str = "data/cache/"

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    attention_type: str = "mha"
    num_kv_heads: Optional[int] = None  # 用于 GQA/MQA

@dataclass
class TrainingConfig:
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 3
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

@dataclass
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int = 512
    block_size: int = 128
    stride: int = 64
    train_subset_ratio: float = 1.0

@dataclass
class WandBConfig:
    enabled: bool = False
    project: str = "gpt-attention-comparison"
    name: Optional[str] = None

@dataclass
class ExperimentConfig:
    name: str
    model_config: ModelConfig
    training_config: TrainingConfig
    data_config: DataConfig
    wandb_config: WandBConfig
    output_dir: str = "outputs"
    seed: int = 42
    task_name: str = "mrpc"  # 用于 GLUE benchmark

class Config:
    def __init__(self, config_path: str = "config/gpt.yaml"):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 加载全局配置
        self.global_config = GlobalConfig(**config_dict.get('global', {}))
        
        # 创建缓存目录
        os.makedirs(self.global_config.cache_dir, exist_ok=True)
        
        # 加载其他配置
        self.model_config = ModelConfig(**config_dict.get('model', {}))
        self.training_config = TrainingConfig(**config_dict.get('training', {}))
        self.data_config = DataConfig(**config_dict.get('data', {}))
        self.wandb_config = WandBConfig(**config_dict.get('wandb', {}))
        
        # 创建实验配置
        experiment_dict = config_dict.get('experiment', {})
        self.experiment_config = ExperimentConfig(
            model_config=self.model_config,
            training_config=self.training_config,
            data_config=self.data_config,
            wandb_config=self.wandb_config,
            **experiment_dict
        )
    
    @property
    def cache_dir(self) -> str:
        return self.global_config.cache_dir

