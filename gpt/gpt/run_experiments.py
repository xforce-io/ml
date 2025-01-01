import os
from gpt.config import Config, ModelConfig, ExperimentConfig
from gpt.env import Env

def main():
    # 加载配置
    config = Config()
    
    # 创建不同的实验配置
    configs = []
    
    # MHA 配置
    mha_config = ExperimentConfig(
        model_config=ModelConfig(**{
            **config.model_config.__dict__,
            "attention_type": "mha"
        }),
        training_config=config.training_config,
        data_config=config.data_config,
        output_dir="outputs/mha",
        use_wandb=config.experiment_config.use_wandb,
        task_name=config.experiment_config.task_name
    )
    configs.append(mha_config)
    
    # GQA 配置
    gqa_config = ExperimentConfig(
        model_config=ModelConfig(**{
            **config.model_config.__dict__,
            "attention_type": "gqa",
            "num_kv_heads": 4
        }),
        training_config=config.training_config,
        data_config=config.data_config,
        output_dir="outputs/gqa",
        use_wandb=config.experiment_config.use_wandb,
        task_name=config.experiment_config.task_name
    )
    configs.append(gqa_config)
    
    # MQA 配置
    mqa_config = ExperimentConfig(
        model_config=ModelConfig(**{
            **config.model_config.__dict__,
            "attention_type": "mqa",
            "num_kv_heads": 1
        }),
        training_config=config.training_config,
        data_config=config.data_config,
        output_dir="outputs/mqa",
        use_wandb=config.experiment_config.use_wandb,
        task_name=config.experiment_config.task_name
    )
    configs.append(mqa_config)
    
    # 运行实验
    env = Env(configs[0])  # 使用第一个配置初始化环境
    env.run_experiments(configs)

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    main() 