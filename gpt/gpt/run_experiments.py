import os
import copy
import wandb
import torch
import numpy as np
import random
from gpt.config import Config, ModelConfig, ExperimentConfig
from gpt.env import Env

def set_seed(seed: int):
    """设置随机种子以确保实验可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def main():
    # 加载配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.experiment_config.seed)
    
    # 创建环境
    env = Env(
        config=config.experiment_config
    )
    
    # 运行实验
    configs = []
    
    # MHA 配置
    mha_config = copy.deepcopy(config.experiment_config)
    mha_config.model_config.attention_type = "mha"
    mha_config.model_config.num_kv_heads = None
    configs.append(mha_config)
    
    # GQA 配置
    gqa_config = copy.deepcopy(config.experiment_config)
    gqa_config.model_config.attention_type = "gqa"
    gqa_config.model_config.num_kv_heads = 4
    configs.append(gqa_config)
    
    # MQA 配置
    mqa_config = copy.deepcopy(config.experiment_config)
    mqa_config.model_config.attention_type = "mqa"
    mqa_config.model_config.num_kv_heads = 1
    configs.append(mqa_config)
    
    # 运行实验
    env.run_experiments(configs)
    
    # 如果启用了 wandb，关闭它
    if config.wandb_config.enabled:
        wandb.finish()

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    main() 