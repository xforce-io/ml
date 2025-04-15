from __future__ import annotations
import traceback
from hex.agents.agent import Experience
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from hex.config import ExitConfig, get_current_device
from hex.hex import Board, State
from hex.log import ERROR, INFO, DEBUG, WARNING
import logging
from typing import List, Tuple, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import requests
import math
from functools import lru_cache

logger = logging.getLogger(__name__)

class HexNet(nn.Module):
    """Hex游戏的神经网络模型"""
    def __init__(
            self, 
            board_size: int, 
            num_channels: int, 
            policy_channels: int = 32,
            value_channels: int = 32,
            dropout_rate: float = 0.1):
        super().__init__()
        self.board_size = board_size
        self.dropout_rate = dropout_rate
        
        # 共享特征提取层
        self.conv1 = nn.Conv2d(3, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2d = nn.Dropout2d(dropout_rate)
        
        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, policy_channels, 1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc1 = nn.Linear(policy_channels * board_size * board_size, 256)
        self.policy_fc2 = nn.Linear(256, board_size * board_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, value_channels, 1)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dropout2d(x)  # 只在最后一层卷积后使用 Dropout2D
        x = F.relu(x)
        
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(-1, self.policy_conv.out_channels * self.board_size * self.board_size)
        policy = self.dropout(policy)
        policy = F.relu(self.policy_fc1(policy))
        policy = self.dropout(policy)
        policy = self.policy_fc2(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(-1, self.value_conv.out_channels * self.board_size * self.board_size)
        value = self.dropout(value)
        value = F.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

    def _preprocess_state(self, state: State) -> torch.Tensor:
        """将状态转换为神经网络输入格式"""
        # 直接创建目标形状的张量，避免多次内存分配
        tensor = torch.zeros(3, self.board_size, self.board_size, dtype=torch.float32)
        
        # 一次性将numpy数组转换为tensor，并重用
        board = torch.from_numpy(state.board.board)
        
        # 使用torch.where进行向量化操作，避免创建中间布尔张量
        tensor[0] = torch.where(board == 1, 1.0, 0.0)
        tensor[1] = torch.where(board == 2, 1.0, 0.0)
        tensor[2] = torch.where(board == 0, 1.0, 0.0)
        
        return tensor

class StateData(BaseModel):
    board: List[List[int]]
    current_player: int

class PredictionResponse(BaseModel):
    action_probs: List[float]
    value: float

class TrainingData(BaseModel):
    states: List[StateData]
    action_probs: List[List[float]]
    rewards: List[float]

class TrainingResponse(BaseModel):
    policy_loss: float
    value_loss: float
    total_loss: float

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    创建带有 linear warmup 和 cosine decay 的学习率调度器
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: warmup 阶段的步数
        total_steps: 总训练步数
    """
    def lr_lambda(current_step: int):
        # warmup 阶段
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # decay 阶段
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

class HexNetTrainerPredictor:
    """神经网络训练和预测服务"""
    def __init__(
            self, 
            config: ExitConfig,
            network: HexNet, 
            device: torch.device):
        self.network = network.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 计算 warmup 步数和总步数
        warmup_steps = int(config.num_steps * 0.1)  # 比如用总步数的 10% 作为 warmup
        total_steps = config.num_steps
        
        # 使用 warmup + cosine decay 调度器
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        self.network.eval()
        self._predict_cache = lru_cache(maxsize=10000)(self._predict_uncached)

    def _predict_uncached(self, zipped: bytes) -> Tuple[np.ndarray, float]:
        """实际执行推理的未缓存方法"""
        return self._predict_state(State.from_zipped(zipped))

    def _predict_state(self, state: State) -> Tuple[np.ndarray, float]:
        """实际执行推理的未缓存方法"""
        with torch.no_grad():
            state_tensor = self.network._preprocess_state(state).unsqueeze(0).to(self.device)
            policy, value = self.network(state_tensor)
            DEBUG(logger, f"Predicted probabilities: {policy.cpu().numpy().flatten()}, value: {value.item()}")
            return policy.cpu().numpy().flatten(), value.item()

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        """带缓存的推理方法"""
        return self._predict_state(state)

    def train(self, batch: List[dict]) -> Tuple[float, float]:
        """训练网络"""
        self.network.train()
        try:
            # 准备训练数据
            states = torch.stack([
                self.network._preprocess_state(exp['state']) 
                for exp in batch
            ]).to(self.device)
            
            target_policies = torch.tensor(
                np.array([exp['action_probs'] for exp in batch]),
                dtype=torch.float32
            ).to(self.device)
            
            target_values = torch.tensor(
                np.array([exp['reward'] for exp in batch]),
                dtype=torch.float32
            ).to(self.device)
            
            # 检查输入数据
            if torch.isnan(states).any() or torch.isnan(target_policies).any() or torch.isnan(target_values).any():
                raise ValueError("输入数据包含 NaN")
            
            # 确保目标策略是概率分布
            assert torch.allclose(target_policies.sum(dim=1), torch.ones_like(target_policies.sum(dim=1))), "目标策略不是有效的概率分布"
            
            # 前向传播
            predicted_policies, predicted_values = self.network(states)
            
            # 计算策略损失
            policy_loss = -(target_policies * predicted_policies).sum(dim=1).mean()
            
            # 计算价值损失
            value_loss = F.mse_loss(predicted_values.squeeze(), target_values)
            
            # 平衡策略和价值学习
            total_loss = policy_loss + value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step()
            
            self._predict_cache.cache_clear()
            
            return policy_loss.item(), value_loss.item()
            
        except Exception as e:
            ERROR(logger, f"训练过程发生错误: {str(e)}")
            raise
        finally:
            self.network.eval()

app = FastAPI()
network_server: HexNetTrainerPredictor = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(state: StateData):
    try:
        board_array = np.array(state.board)
        state_obj = State(
            board=Board(board=board_array), 
            current_player=state.current_player)
        action_probs, value = network_server.predict(state_obj)
        return {
            "action_probs": action_probs.tolist(),
            "value": float(value)
        }
    except Exception as e:
        ERROR(logger, f"Prediction error: {e} traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train(data: TrainingData):
    try:
        batch = []
        for state, probs, reward in zip(data.states, data.action_probs, data.rewards):
            board_array = np.array(state.board)
            batch.append({
                'state': State(Board(board=board_array), state.current_player),
                'action_probs': np.array(probs),
                'reward': reward
            })
        policy_loss, value_loss = network_server.train(batch)
        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "total_loss": float(policy_loss + value_loss)
        }
    except Exception as e:
        ERROR(logger, f"Training error: {e} traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
async def save_model(path: str):
    try:
        network_server.network.save_model(path)
        return {"status": "success"}
    except Exception as e:
        ERROR(logger, f"Save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load")
async def load_model(path: str):
    try:
        network_server.network.load_model(path)
        return {"status": "success"}
    except Exception as e:
        ERROR(logger, f"Load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_server(
        config: ExitConfig, 
        board_size: int, 
        host: str = "127.0.0.1", 
        port: int = 8000):
    """启动神经网络服务器"""
    global network_server
    
    try:
        # 初始化设备
        device = get_current_device()
        INFO(logger, f"Using device: {device}")
        
        # 创建网络
        network = HexNet(
            board_size=board_size,
            num_channels=config.num_channels,
            policy_channels=config.policy_channels,
            value_channels=config.value_channels
        ).to(device)
        
        # 创建服务器
        network_server = HexNetTrainerPredictor(config, network, device)
        
        # 启动FastAPI服务器
        INFO(logger, f"Starting server at {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning")
        
    except Exception as e:
        ERROR(logger, f"Server startup error: {e}")
        raise

class NetworkClient:
    """基于HTTP的神经网络客户端"""
    def __init__(self, config: ExitConfig, port: Optional[int] = None):
        self.config = config
        self.port = port or config.network_server_port
        self.base_url = f"http://127.0.0.1:{self.port}"
        
        # 使用连接池来复用连接
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,  # 连接池大小
            pool_maxsize=100,     # 最大连接数
            max_retries=3,        # 重试次数
            pool_block=True       # 连接池满时阻塞而不是抛出错误
        )
        self.session.mount('http://', adapter)

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={
                    "board": state.board.tolist(),
                    "current_player": state.current_player
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result["action_probs"]), result["value"]
        except Exception as e:
            ERROR(logger, f"预测错误: {e}")
            raise

    def train(self, batch: List[Experience]) -> Tuple[float, float]:
        """请求训练"""
        try:
            data = {
                "states": [
                    {
                        "board": exp.state.board.board.tolist(),
                        "current_player": exp.state.current_player
                    }
                    for exp in batch
                ],
                "action_probs": [exp.export_probs.tolist() for exp in batch],
                "rewards": [exp.reward for exp in batch]
            }
            response = self.session.post(f"{self.base_url}/train", json=data)
            response.raise_for_status()
            result = response.json()
            return result["policy_loss"], result["value_loss"]  # 返回两个损失值
        except Exception as e:
            ERROR(logger, f"训练错误: {e} traceback: {traceback.format_exc()}")
            raise

    def save(self, path: str) -> Dict[str, str]:
        """请求保存模型"""
        response = self.session.post(f"{self.base_url}/save", params={"path": path})
        response.raise_for_status()
        return response.json()

    def load(self, path: str) -> Dict[str, str]:
        """请求加载模型"""
        response = self.session.post(f"{self.base_url}/load", params={"path": path})
        response.raise_for_status()
        return response.json()

class HexNetWrapper:
    """神经网络服务器包装器"""
    def __init__(self, config: ExitConfig, board_size: int):
        self.config = config
        self.board_size = board_size
        self.network_client = None
        self._server_thread = None
        self._server_started = False
        self._port = None  # 添加端口属性
        
    def __getstate__(self):
        """自定义序列化行为"""
        state = self.__dict__.copy()
        # 移除不可序列化的对象
        state['_server_thread'] = None
        state['network_client'] = None
        # 保留端口信息，这样新进程可以知道使用哪个端口
        return state

    def __setstate__(self, state):
        """自定义反序列化行为"""
        self.__dict__.update(state)
        # 不要在这里启动服务器，让用户显式调用start()
        self._server_thread = None
        self.network_client = None
        self._server_started = False

    def start(self):
        """启动服务器和客户端"""
        if self._server_started:
            return
            
        INFO(logger, "Initializing network server...")
        
        # 如果没有指定端口，则动态分配一个
        if not self._port:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                self._port = s.getsockname()[1]
                
        # 使用实例特定的端口
        server_port = self._port
        
        # 创建并启动服务器线程
        self._server_thread = threading.Thread(
            target=run_server,
            kwargs={
                "config": self.config, 
                "board_size": self.board_size,
                "host": self.config.network_server_host,
                "port": server_port
            },
            daemon=True
        )
        self._server_thread.start()
        INFO(logger, f"Server started at {self.config.network_server_host}:{server_port}")
        
        # 创建客户端
        self.network_client = NetworkClient(self.config, port=server_port)
        
        # 等待服务器启动并进行健康检查
        import time
        max_retries = 5
        retry_interval = 1.0
        
        time.sleep(retry_interval)

        for i in range(max_retries):
            try:
                response = requests.get(
                    f"http://{self.config.network_server_host}:{server_port}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    self._server_started = True
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if i < max_retries - 1:
                    WARNING(logger, f"Failed to connect to server, retrying in {retry_interval}s...")
                    time.sleep(retry_interval)
                else:
                    ERROR(logger, "Failed to connect to network server after multiple attempts")
                    raise RuntimeError("Network server initialization failed")
    
    def start_client(self):
        if self.network_client is None:
            self.network_client = NetworkClient(self.config, port=self._port)

    def clone(self):
        """克隆网络客户端"""
        new_wrapper = HexNetWrapper(self.config, self.board_size)
        new_wrapper._port = self._port  # 共享相同的端口
        new_wrapper.network_client = NetworkClient(self.config, port=self._port)
        new_wrapper._server_started = True  # 标记为已启动，因为我们共享同一个服务器
        return new_wrapper

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        """请求推理结果"""
        return self.network_client.predict(state)

    def train(self, batch: List[Experience]) -> Tuple[float, float]:
        """请求训练"""
        return self.network_client.train(batch)

    def save(self, path: str) -> Dict[str, str]:
        """请求保存模型"""
        return self.network_client.save(path)

    def load(self, path: str) -> Dict[str, str]:
        """请求加载模型"""
        return self.network_client.load(path)

# 添加健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "ok"}