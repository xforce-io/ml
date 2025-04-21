import random
import numpy as np
import os
from .base_algo import Algo

class AlgoRandom(Algo):
    """随机动作算法实现"""
    def __init__(self, env, config, device):
        super().__init__(env, config, device)
        
        # 由于随机算法不需要学习，所以不需要额外的配置参数
        
    def selectAction(self, state, deterministic=False):
        """无论状态如何，都返回随机动作"""
        return np.random.randint(self.num_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """随机算法不学习，仅记录步数"""
        # 实际上，算法不需要学习，所以这里什么也不做
        # 但是我们在 'updateStepsDone' 方法中更新步数
        return None
    
    def save(self, directory, filename):
        """保存模型 (随机算法无需保存模型)"""
        super().save(directory, filename) # 确保目录存在
        return True
    
    def load(self, filepath):
        """加载模型 (随机算法无需加载模型)"""
        return True
    
    def setEvalMode(self):
        """设置为评估模式 (随机算法无特殊模式)"""
        pass
    
    def setTrainMode(self):
        """设置为训练模式 (随机算法无特殊模式)"""
        pass 