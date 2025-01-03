from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler

from gpt.config import Config, ExperimentConfig
from gpt.benchmark import BenchmarkGLUE
from gpt.gpt import GPT
import wandb
import os
import numpy as np
from gpt.model_cache import ModelCacheManager

class Env:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_cache = ModelCacheManager()
        
        # 检测并设置设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("使用 MPS (Metal Performance Shaders) 设备加速训练")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("使用 CUDA 设备加速训练")
        else:
            self.device = torch.device("cpu")
            print("使用 CPU 设备训练")
        
        # 获取全局配置
        self.global_config = Config().global_config
        
        # 初始化数据集
        if config.data_config.dataset_name == "wikitext":
            self.dataset = load_dataset(
                path=config.data_config.dataset_name, 
                name=config.data_config.dataset_config,
                cache_dir=os.path.join(self.global_config.cache_dir, "datasets"),
            )
        else:
            self.dataset = load_dataset(
                path=config.data_config.dataset_name,
                name=config.data_config.dataset_config,
                cache_dir=os.path.join(self.global_config.cache_dir, "datasets"),
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=os.path.join(self.global_config.cache_dir, "tokenizers"),
            local_files_only=True  # 只使用本地缓存的文件
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
    def preprocess_function(self, examples):
        # 对文本进行编码
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.data_config.max_length,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=self.config.data_config.stride,
            return_length=True,
        )
        
        # 创建用于语言建模的输入-输出对
        input_batch = []
        label_batch = []
        for length, input_ids in zip(tokenized["length"], tokenized["input_ids"]):
            if length > self.config.data_config.block_size:
                input_batch.append(input_ids[:self.config.data_config.block_size])
                label_batch.append(input_ids[1:self.config.data_config.block_size+1])
            else:
                input_batch.append(input_ids)
                label_batch.append(input_ids[1:] + [self.tokenizer.pad_token_id])
                
        result = {
            "input_ids": input_batch,
            "labels": label_batch,
            "attention_mask": [[1] * len(input_ids) for input_ids in input_batch]
        }
        
        # 将列表转换为 numpy 数组
        result = {k: np.array(v) for k, v in result.items()}
        return result
    
    def train_model(self) -> GPT:
        # 检查缓存
        cached_model_path = self.model_cache.get_cache_path(
            self.config.model_config,
            self.config.training_config,
            self.config.data_config
        )
        
        if cached_model_path:
            print(f"找到缓存模型，从 {cached_model_path} 加载")
            return GPT.from_pretrained(cached_model_path, device=self.device)
        
        print("未找到缓存模型，开始训练...")
        
        # 初始化模型
        model = GPT(self.config.model_config).to(self.device)
        
        # 预处理数据集
        processed_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset.column_names["train"],
            load_from_cache_file=False
        )
        
        # 采样训练数据
        if hasattr(self.config.data_config, 'train_subset_ratio') and self.config.data_config.train_subset_ratio < 1.0:
            train_size = len(processed_datasets["train"])
            subset_size = int(train_size * self.config.data_config.train_subset_ratio)
            # 随机采样，但固定随机种子以保证可复现性
            rng = np.random.default_rng(42)
            subset_indices = rng.choice(train_size, size=subset_size, replace=False)
            processed_datasets["train"] = processed_datasets["train"].select(subset_indices)
            print(f"使用 {self.config.data_config.train_subset_ratio:.1%} 的训练数据进行训练 "
                  f"({subset_size}/{train_size})")
        
        # 设置数据集格式
        processed_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            processed_datasets["train"],
            batch_size=self.config.training_config.train_batch_size,
            shuffle=True
        )
        
        if "validation" in processed_datasets:
            val_dataloader = DataLoader(
                processed_datasets["validation"],
                batch_size=self.config.training_config.eval_batch_size,
                shuffle=False
            )
        else:
            val_dataloader = None
        
        # 优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.training_config.learning_rate
        )
        num_training_steps = len(train_dataloader) * self.config.training_config.num_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.training_config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 初始化 wandb
        if self.config.wandb_config.enabled:
            wandb.init(
                project=self.config.wandb_config.project,
                name=self.config.wandb_config.name,
                config=self.config.__dict__
            )
        
        # 训练循环
        for epoch in range(self.config.training_config.num_epochs):
            # 使用 GPT 类的 train_epoch 方法
            metrics = model.train_epoch(
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                epoch=epoch,
                val_dataloader=val_dataloader
            )
            
            # 更新学习率
            lr_scheduler.step()
            
            # 记录到 wandb
            if self.config.wandb_config.enabled:
                wandb.log({
                    "epoch": epoch,
                    **metrics
                })
        
        # 保存模型
        model_save_path = os.path.join(
            self.config.output_dir, 
            f"{self.config.model_config.attention_type}"
        )
        model.save_pretrained(model_save_path)
        
        # 将模型保存到缓存
        self.model_cache.save_model_to_cache(
            model_path=model_save_path,
            model_config=self.config.model_config,
            training_config=self.config.training_config,
            data_config=self.config.data_config
        )
        
        return model
    
    def run_experiments(self, configs: List[ExperimentConfig]):
        """运行多个实验并比较结果"""
        benchmark = BenchmarkGLUE(task_name=self.config.task_name)
        models = []
        
        for config in configs:
            self.config = config
            model = self.train_model()
            models.append(model)
            benchmark.run_benchmark(model, batch_size=config.training_config.eval_batch_size)
        
        # 生成对比图表
        benchmark.plot_results(
            save_path=os.path.join(self.config.output_dir, "comparison.png")
        )
        benchmark.print_results() 