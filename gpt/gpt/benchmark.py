import os
import torch
import time
import numpy as np
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from datasets import load_dataset, load_from_disk, DatasetBuilder
from datasets.builder import DatasetBuilder
from transformers import AutoTokenizer
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from .config import Config
import evaluate
import glob

class BenchmarkBase:
    def __init__(self):
        config = Config()
        self.global_config = config.global_config
        self.results = {}
    
    def reset_memory_stats(self):
        """重置GPU内存统计"""
        if torch.cuda.is_available():
            reset_peak_memory_stats()
    
    def measure_memory(self) -> float:
        """测量GPU内存使用"""
        if torch.cuda.is_available():
            return max_memory_allocated() / 1024**3
        return 0.0
    
    def measure_cpu_memory(self) -> float:
        """测量CPU内存使用"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    
    def log_metrics(self, model_type: str, metrics: Dict[str, Any]):
        """记录性能指标"""
        self.results[model_type] = metrics
    
    def plot_results(self, save_path: str = None):
        """绘制性能对比图表"""
        metrics_data = {
            "Inference Time (s)": [],
            "GPU Memory (GB)": [],
            "CPU Memory (GB)": [],
            "Model Type": []
        }
        
        for model_type, metrics in self.results.items():
            metrics_data["Inference Time (s)"].append(float(metrics["avg_inference_time"][:-1]))
            metrics_data["GPU Memory (GB)"].append(float(metrics["gpu_memory_usage"][:-2]))
            metrics_data["CPU Memory (GB)"].append(float(metrics["cpu_memory_usage"][:-2]))
            metrics_data["Model Type"].append(model_type)
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 绘制柱状图
        for i, metric in enumerate(["Inference Time (s)", "GPU Memory (GB)", "CPU Memory (GB)"]):
            sns.barplot(x="Model Type", y=metric, data=metrics_data, ax=axes[i])
            axes[i].set_title(metric)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class BenchmarkGLUE(BenchmarkBase):
    def __init__(self, task_name: str = "mrpc"):
        super().__init__()
        self.task_name = task_name
        
        # 检查缓存目录
        cache_dir = os.path.join(self.global_config.cache_dir, "datasets")
        
        t0 = time.time()
        self.dataset = load_dataset("glue", task_name, cache_dir=cache_dir)
        t1 = time.time()
        self.metric = evaluate.load("glue", self.task_name, cache_dir=cache_dir)
        t2 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=self.global_config.cache_dir)
        t3 = time.time()
        print(f"loading dataset cost {t1 - t0:.2f} seconds, loading metric cost {t2 - t1:.2f} seconds, loading tokenizer cost {t3 - t2:.2f} seconds")
        
    def preprocess_function(self, examples):
        # 根据任务类型处理数据
        if self.task_name == "mrpc":
            texts = ((examples["sentence1"], examples["sentence2"]))
        else:
            texts = (examples["sentence"],)
            
        result = self.tokenizer(*texts, padding=True, truncation=True, max_length=512)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result
        
    def run_benchmark(self, model, batch_size: int = 32, num_batches: int = None):
        device = next(model.parameters()).device
        model.eval()
        
        # 预处理验证集
        eval_dataset = self.dataset["validation"].map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["validation"].column_names
        )
        
        # 记录开始时间和内存
        start_time = time.time()
        self.reset_memory_stats()
        
        all_predictions = []
        all_labels = []
        
        # 评估
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset[i:i + batch_size]
            inputs = {k: torch.tensor(v).to(device) for k, v in batch.items() 
                     if k != "labels"}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            predictions = outputs["logits"].argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            if "labels" in batch:
                all_labels.extend(batch["labels"])
        
        # 计算指标
        metrics = self.metric.compute(predictions=all_predictions, 
                                    references=all_labels)
        
        # 添加性能指标
        metrics.update({
            "avg_inference_time": f"{(time.time() - start_time) / len(eval_dataset):.4f}s",
            "gpu_memory_usage": f"{self.measure_memory():.2f}GB",
            "cpu_memory_usage": f"{self.measure_cpu_memory():.2f}GB",
        })
        
        self.log_metrics(f"{model.config.attention_type.upper()}", metrics) 