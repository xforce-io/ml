import os
import copy
from gpt.config import Config
import wandb
import torch
import torch.distributed as dist
import numpy as np
import random
from env import Env

def setup_distributed(rank: int, world_size: int):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

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
    # 检查是否在分布式环境中运行
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank != -1:
        # 分布式训练设置
        setup_distributed(local_rank, world_size)
    
    # 加载配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.experiment_config.seed + local_rank)  # 每个进程使用不同的种子
    
    # 创建环境
    env = Env(
        config=config.experiment_config,
        local_rank=local_rank,
        world_size=world_size
    )
    
    # 运行实验
    configs = []
    
    # MHA 配置
    mha_config = copy.deepcopy(config.experiment_config)
    mha_config.model_config.attention_type = "mha"
    mha_config.model_config.num_kv_heads = None
    mha_config.name = "mha"
    configs.append(mha_config)
    
    # GQA 配置
    gqa_config = copy.deepcopy(config.experiment_config)
    gqa_config.model_config.attention_type = "gqa"
    gqa_config.model_config.num_kv_heads = 4
    gqa_config.name = "gqa"
    configs.append(gqa_config)
    
    # MQA 配置
    mqa_config = copy.deepcopy(config.experiment_config)
    mqa_config.model_config.attention_type = "mqa"
    mqa_config.model_config.num_kv_heads = 1
    mqa_config.name = "mqa"
    configs.append(mqa_config)
    
    # 运行实验
    env.run_experiments(configs)
    
    # 如果启用了 wandb，关闭它
    if config.wandb_config.enabled and local_rank in [-1, 0]:
        wandb.finish()
    
    # 清理分布式环境
    if local_rank != -1:
        cleanup_distributed()

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    main() 