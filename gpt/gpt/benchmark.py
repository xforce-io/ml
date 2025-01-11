import os
import torch
import time
import numpy as np
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from datasets import load_dataset, load_from_disk, DatasetBuilder
from datasets.builder import DatasetBuilder
from transformers import AutoTokenizer
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from .config import Config
import evaluate
import glob
from tqdm import tqdm
import torch.nn as nn
import random

class BenchmarkBase:
    """基准测试的基类，提供基础的性能测量功能"""
    def __init__(self):
        config = Config()
        self.global_config = config.global_config
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def reset_memory_stats(self) -> None:
        """重置GPU内存统计"""
        if torch.cuda.is_available():
            reset_peak_memory_stats()
    
    def measure_memory(self) -> float:
        """测量GPU内存使用（GB）"""
        if torch.cuda.is_available():
            return max_memory_allocated() / 1024**3
        return 0.0
    
    def measure_cpu_memory(self) -> float:
        """测量CPU内存使用（GB）"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    
    def log_metrics(self, model_type: str, metrics: Dict[str, Any]) -> None:
        """记录性能指标"""
        self.results[model_type] = metrics
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """绘制性能对比图表"""
        if not self.results:
            print("没有可用的结果数据进行绘图")
            return
            
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

    def print_results_as_markdown(self) -> None:
        """打印结果为 markdown 表格"""
        if not self.results:
            print("没有可用的结果数据")
            return
            
        print("| Model Type | Inference Time (s) | GPU Memory (GB) | Accuracy |")
        print("| --- | --- | --- | --- |")
        for model_type, metrics in self.results.items():
            print(f"| {model_type} | {metrics['avg_inference_time']} | {metrics['peak_gpu_memory']} | {metrics['accuracy']} |")

class BenchmarkGLUE(BenchmarkBase):
    """GLUE 基准测试的实现"""
    SUPPORTED_TASKS = {
        "mrpc": {"description": "句子对等价性判断", "suffix": "答案："},
        "sst2": {"description": "情感分析", "suffix": "情感："},
        "cola": {"description": "语法可接受性判断", "suffix": "语法："}
    }
    
    def __init__(self, task_name: str = "mrpc", num_shots: int = 8):
        """初始化 GLUE 基准测试
        
        Args:
            task_name: GLUE 任务名称，支持 "mrpc", "sst2", "cola"
            num_shots: few-shot 学习的样本数量
        """
        super().__init__()
        if task_name not in self.SUPPORTED_TASKS:
            raise ValueError(f"不支持的任务类型：{task_name}。支持的任务：{list(self.SUPPORTED_TASKS.keys())}")
            
        self.task_name = task_name
        self.num_shots = num_shots
        
        # 检查缓存目录
        cache_dir = os.path.join(self.global_config.cache_dir, "datasets")
        metric_cache_dir = os.path.join(self.global_config.cache_dir, "metrics")
        os.makedirs(metric_cache_dir, exist_ok=True)
        
        # 加载数据集和评估指标
        self._load_dataset_and_metric(cache_dir, metric_cache_dir)
        
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=self.global_config.cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_dataset_and_metric(self, cache_dir: str, metric_cache_dir: str) -> None:
        """加载数据集和评估指标"""
        t0 = time.time()
        self.dataset = load_dataset("glue", self.task_name, cache_dir=cache_dir)
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
                    print(f"加载评估指标失败（{max_retries}次尝试）：{e}")
                    raise
                print(f"第{attempt + 1}次尝试失败，正在重试...")
                time.sleep(1)
        
        t2 = time.time()
        print(f"加载数据集耗时：{t1 - t0:.2f}秒")
        print(f"加载评估指标耗时：{t2 - t1:.2f}秒")

    def _prepare_few_shot_prompt(self, examples: List[Dict[str, Any]], test_example: Dict[str, Any]) -> str:
        """准备 few-shot prompt
        
        Args:
            examples: few-shot 示例列表
            test_example: 待测试的样本
            
        Returns:
            格式化的 prompt 字符串
        """
        task_info = self.SUPPORTED_TASKS[self.task_name]
        
        if self.task_name == "mrpc":
            prompt = f"判断以下句子对是否等价。如果等价输出1，不等价输出0。\n\n"
            
            # 添加示例
            for example in examples:
                prompt += f"句子1：{example['sentence1']}\n"
                prompt += f"句子2：{example['sentence2']}\n"
                prompt += f"答案：{example['label']}\n\n"
            
            # 添加测试样本
            prompt += f"句子1：{test_example['sentence1']}\n"
            prompt += f"句子2：{test_example['sentence2']}\n"
            prompt += "答案："
            
        elif self.task_name == "sst2":
            prompt = f"判断以下电影评论的情感倾向。如果是正面情感输出1，负面情感输出0。\n\n"
            
            # 添加示例
            for example in examples:
                prompt += f"评论：{example['sentence']}\n"
                prompt += f"情感：{example['label']}\n\n"
            
            # 添加测试样本
            prompt += f"评论：{test_example['sentence']}\n"
            prompt += "情感："
            
        elif self.task_name == "cola":
            prompt = f"判断以下句子在语法上是否正确。如果语法正确输出1，语法错误输出0。\n\n"
            
            # 添加示例
            for example in examples:
                prompt += f"句子：{example['sentence']}\n"
                prompt += f"语法：{example['label']}\n\n"
            
            # 添加测试样本
            prompt += f"句子：{test_example['sentence']}\n"
            prompt += "语法："
            
        return prompt

    def _process_model_output(self, output_text: str, prompt_suffix: str) -> int:
        """处理模型输出，提取预测标签"""
        if prompt_suffix in output_text:
            label_text = output_text.split(prompt_suffix)[-1].strip()
            # 添加日志以便调试
            print(f"Raw output: {output_text}")
            print(f"Extracted label text: {label_text}")
            try:
                label = int(label_text)
                if label in [0, 1]:
                    return label
                else:
                    print(f"Invalid label value: {label}")
            except ValueError:
                print(f"Failed to parse label: {label_text}")
        else:
            print(f"Prompt suffix '{prompt_suffix}' not found in output[{output_text}]")
        
        # 随机返回标签而不是总是返回0
        return random.choice([0, 1])

    def run_benchmark(
            self, 
            experiment_name: str,
            model: nn.Module, 
            batch_size: int = 32) -> None:
        """运行基准测试
        
        Args:
            experiment_name: 实验名称
            model: 要评估的模型
            batch_size: 批处理大小
        """
        device = next(model.parameters()).device
        model.eval()
        
        # 获取原始模型（如果是 DDP 模型）
        base_model = model.module if hasattr(model, 'module') else model
        
        # 准备 few-shot 样本
        train_dataset = self.dataset["train"]
        rng = np.random.default_rng(42)
        shot_indices = rng.choice(len(train_dataset), size=self.num_shots, replace=False)
        few_shot_examples = [dict(train_dataset[int(i)]) for i in shot_indices]
        
        print(f"\n使用 {self.num_shots} 个训练样本进行 few-shot 评估")
        
        # 评估验证集
        eval_dataset = self.dataset["validation"]
        all_predictions = []
        all_labels = []
        
        # 重置内存统计
        self.reset_memory_stats()
        start_time = time.time()
        initial_memory = self.measure_memory()
        
        prompt_suffix = self.SUPPORTED_TASKS[self.task_name]["suffix"]
        
        try:
            # 使用进度条
            progress_bar = tqdm(
                range(0, len(eval_dataset), batch_size),
                desc="Few-shot Evaluating",
                ncols=100
            )
            
            for i in progress_bar:
                batch_examples = [
                    dict(eval_dataset[j]) 
                    for j in range(i, min(i + batch_size, len(eval_dataset)))
                ]
                
                # 准备输入
                batch_prompts = [
                    self._prepare_few_shot_prompt(few_shot_examples, example)
                    for example in batch_examples
                ]
                
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                # 生成预测
                with torch.no_grad():
                    suffix_inputs = self.tokenizer(
                        [prompt_suffix],
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).to(device)
                    
                    outputs = base_model.generate(
                        **inputs,
                        max_new_tokens=len(suffix_inputs["input_ids"][0]) + 1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,
                        num_beams=1
                    )
                    
                    # 处理输出
                    predictions = []
                    for output in outputs:
                        generated_text = self.tokenizer.decode(output[-10:], skip_special_tokens=True)
                        label = self._process_model_output(generated_text, prompt_suffix)
                        predictions.append(label)
                
                all_predictions.extend(predictions)
                all_labels.extend([example["label"] for example in batch_examples])
                
                # 更新进度条
                progress_bar.set_postfix({
                    "processed": f"{min(i + batch_size, len(eval_dataset))}/{len(eval_dataset)}"
                })
                
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
        
        self.log_metrics(experiment_name, metrics) 
