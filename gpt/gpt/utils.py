import logging
import torch
from typing import Dict, Any

def setup_logging(name: str = "gpt", level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def compute_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """计算模型大小和参数数量"""
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return {
        "num_parameters": num_params,
        "model_size_mb": model_size_mb
    } 