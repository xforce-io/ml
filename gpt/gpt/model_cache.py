import os
import json
import hashlib
from typing import Dict, Optional
from dataclasses import asdict
from gpt.config import ModelConfig, TrainingConfig, DataConfig

class ModelCacheManager:
    def __init__(self, cache_dir: str = "data/cache/trained"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index_path = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict:
        """加载缓存索引"""
        if os.path.exists(self.cache_index_path):
            with open(self.cache_index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _generate_cache_key(self, 
                          model_config: ModelConfig,
                          training_config: TrainingConfig,
                          data_config: DataConfig) -> str:
        """生成缓存键
        
        基于模型配置、训练配置和数据配置生成唯一的缓存键。
        只包含影响模型性能的关键参数。
        """
        key_dict = {
            # 模型架构参数
            "model": {
                "hidden_size": model_config.hidden_size,
                "num_layers": model_config.num_layers,
                "num_heads": model_config.num_heads,
                "attention_type": model_config.attention_type,
                "num_kv_heads": model_config.num_kv_heads,
            },
            # 训练参数
            "training": {
                "num_epochs": training_config.num_epochs,
                "learning_rate": training_config.learning_rate,
                "weight_decay": training_config.weight_decay,
            },
            # 数据集参数
            "data": {
                "dataset_name": data_config.dataset_name,
                "dataset_config": data_config.dataset_config,
                "train_subset_ratio": data_config.train_subset_ratio,
            }
        }
        
        # 将字典转换为规范化的字符串并计算哈希
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_path(self, 
                      model_config: ModelConfig,
                      training_config: TrainingConfig,
                      data_config: DataConfig) -> Optional[str]:
        """获取缓存模型路径
        
        如果缓存存在，返回路径；否则返回 None
        """
        cache_key = self._generate_cache_key(model_config, training_config, data_config)
        cache_info = self.cache_index.get(cache_key)
        
        if cache_info and os.path.exists(cache_info["path"]):
            return cache_info["path"]
        return None
    
    def save_model_to_cache(self,
                           model_path: str,
                           model_config: ModelConfig,
                           training_config: TrainingConfig,
                           data_config: DataConfig):
        """将模型保存到缓存
        
        Args:
            model_path: 模型文件所在路径
            model_config: 模型配置
            training_config: 训练配置
            data_config: 数据配置
        """
        cache_key = self._generate_cache_key(model_config, training_config, data_config)
        cache_model_dir = os.path.join(self.cache_dir, cache_key)
        
        # 复制模型文件到缓存目录
        os.makedirs(cache_model_dir, exist_ok=True)
        os.system(f"cp -r {model_path}/* {cache_model_dir}/")
        
        # 更新缓存索引
        self.cache_index[cache_key] = {
            "path": cache_model_dir,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "data_config": asdict(data_config),
            "original_path": model_path
        }
        self._save_cache_index()
    
    def clear_cache(self):
        """清除所有缓存"""
        for cache_info in self.cache_index.values():
            if os.path.exists(cache_info["path"]):
                os.system(f"rm -rf {cache_info['path']}")
        self.cache_index = {}
        self._save_cache_index() 