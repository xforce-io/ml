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
from tqdm import tqdm

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
            "Model Type": [],
            "Accuracy": []
        }
        
        for model_type, metrics in self.results.items():
            metrics_data["Inference Time (s)"].append(float(metrics["avg_inference_time"][:-1]))
            metrics_data["GPU Memory (GB)"].append(float(metrics["peak_gpu_memory"][:-2]))
            metrics_data["Model Type"].append(model_type)
            metrics_data["Accuracy"].append(float(metrics["accuracy"]))

        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制柱状图
        for i, metric in enumerate(["Inference Time (s)", "GPU Memory (GB)"]):
            sns.barplot(x="Model Type", y=metric, data=metrics_data, ax=axes[i])
            axes[i].set_title(metric)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def print_results_as_markdown(self):
        """打印结果为 markdown 表格"""
        print("| Model Type | Inference Time (s) | GPU Memory (GB) | Accuracy |")
        print("| --- | --- | --- | --- |")
        for model_type, metrics in self.results.items():
            print(f"| {model_type} | {metrics['avg_inference_time']} | {metrics['peak_gpu_memory']} | {metrics['accuracy']} |")

class BenchmarkGLUE(BenchmarkBase):
    def __init__(self, task_name: str = "mrpc", num_shots: int = 8):
        super().__init__()
        self.task_name = task_name
        self.num_shots = num_shots  # few-shot 样本数量
        
        # 检查缓存目录
        cache_dir = os.path.join(self.global_config.cache_dir, "datasets")
        metric_cache_dir = os.path.join(self.global_config.cache_dir, "metrics")
        os.makedirs(metric_cache_dir, exist_ok=True)
        
        t0 = time.time()
        self.dataset = load_dataset("glue", task_name, cache_dir=cache_dir)
        t1 = time.time()
        
        # 添加重试逻辑和超时设置
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.metric = evaluate.load(
                    "glue",
                    self.task_name,
                    cache_dir=metric_cache_dir,
                    download_mode="force_redownload" if attempt > 0 else None
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to load metric after {max_retries} attempts: {e}")
                    # 使用备用评估方法
                    self.metric = self._get_backup_metric()
                else:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
        
        t2 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=self.global_config.cache_dir
        )
        # 设置 padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        t3 = time.time()
        print(f"loading dataset cost {t1 - t0:.2f} seconds, loading metric cost {t2 - t1:.2f} seconds, loading tokenizer cost {t3 - t2:.2f} seconds")

    def _prepare_few_shot_prompt(self, examples, test_example):
        """准备 few-shot prompt
        
        将训练集中的示例和测试样本组合成 prompt
        """
        prompt = "Determine if the following sentence pairs are equivalent (1) or not equivalent (0).\n\n"
        
        # 添加示例
        for i, example in enumerate(examples):
            if self.task_name == "mrpc":
                prompt += f"Sentence 1: {example['sentence1']}\n"
                prompt += f"Sentence 2: {example['sentence2']}\n"
                prompt += f"Label: {example['label']}\n\n"
            else:
                prompt += f"Text: {example['sentence']}\n"
                prompt += f"Label: {example['label']}\n\n"
        
        # 添加测试样本
        if self.task_name == "mrpc":
            prompt += f"Sentence 1: {test_example['sentence1']}\n"
            prompt += f"Sentence 2: {test_example['sentence2']}\n"
            prompt += "Label:"
        else:
            prompt += f"Text: {test_example['sentence']}\n"
            prompt += "Label:"
        
        return prompt

    def run_benchmark(self, model, batch_size: int = 32):
        device = next(model.parameters()).device
        model.eval()
        
        # 获取原始模型（如果是 DDP 模型）
        if hasattr(model, 'module'):
            base_model = model.module
        else:
            base_model = model
        
        # 从训练集中随机选择 few-shot 样本
        train_dataset = self.dataset["train"]
        rng = np.random.default_rng(42)  # 固定随机种子
        shot_indices = rng.choice(len(train_dataset), size=self.num_shots, replace=False)
        # 将 numpy.int64 转换为 Python int
        shot_indices = [int(i) for i in shot_indices]
        few_shot_examples = [dict(train_dataset[i]) for i in shot_indices]
        
        print(f"\n使用 {self.num_shots} 个训练样本进行 few-shot 评估")
        
        # 评估验证集
        eval_dataset = self.dataset["validation"]
        all_predictions = []
        all_labels = []
        
        # 清理 GPU 缓存并重置内存统计
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 记录开始时间和内存
        start_time = time.time()
        initial_memory = self.measure_memory()
        
        # 使用进度条
        progress_bar = tqdm(
            range(0, len(eval_dataset), batch_size),
            desc="Few-shot Evaluating",
            ncols=100
        )
        
        # 准备标签 token
        label_tokens = ["0", "1"]  # MRPC 任务的标签
        label_token_ids = [
            self.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in label_tokens
        ]
        
        try:
            for i in progress_bar:
                batch_examples = [dict(eval_dataset[j]) for j in range(i, min(i + batch_size, len(eval_dataset)))]
                batch_prompts = [
                    self._prepare_few_shot_prompt(few_shot_examples, example)
                    for example in batch_examples
                ]
                
                # 编码输入
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                # 生成预测
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs["logits"]
                    
                    # 只在标签 token 中选择
                    last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                    label_logits = last_token_logits[:, label_token_ids]  # [batch_size, num_labels]
                    predictions = label_logits.argmax(dim=-1)  # [batch_size]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend([example["label"] for example in batch_examples])
                
                # 更新进度条
                progress_bar.set_postfix({
                    "processed": f"{min(i + batch_size, len(eval_dataset))}/{len(eval_dataset)}"
                })
                
                # 定期清理缓存
                if torch.cuda.is_available() and (i + 1) % (10 * batch_size) == 0:
                    torch.cuda.empty_cache()
        
        finally:
            # 确保测量最终内存使用
            final_memory = self.measure_memory()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 计算指标
        metrics = self.metric.compute(
            predictions=all_predictions,
            references=all_labels
        )
        
        # 添加性能指标
        metrics.update({
            "avg_inference_time": f"{(time.time() - start_time) / len(eval_dataset):.4f}s",
            "peak_gpu_memory": f"{final_memory:.2f}GB",
            "gpu_memory_increase": f"{(final_memory - initial_memory):.2f}GB",
            "num_shots": self.num_shots
        })
        
        # 使用原始模型的配置
        self.log_metrics(f"{base_model.config.attention_type.upper()}", metrics) 