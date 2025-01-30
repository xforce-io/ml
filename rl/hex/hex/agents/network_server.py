from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hex.config import ExitConfig, get_current_device
from hex.hex import State
from hex.log import ERROR, INFO, DEBUG, WARNING
import logging
from typing import List, Tuple, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import requests

logger = logging.getLogger(__name__)

class HexNet(nn.Module):
    """Hex游戏的神经网络模型"""
    def __init__(self, board_size: int, num_channels: int, policy_channels: int = 32):
        super().__init__()
        self.board_size = board_size
        
        # 共享特征提取层
        self.conv1 = nn.Conv2d(3, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        
        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, policy_channels, 1)
        self.policy_fc = nn.Linear(policy_channels * board_size * board_size, 
                                 board_size * board_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, policy_channels, 1)
        self.value_fc1 = nn.Linear(policy_channels * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 策略头
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 32 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # 价值头
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 32 * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

    def _preprocess_state(self, state: State) -> torch.Tensor:
        """将状态转换为神经网络输入格式"""
        current_player = state.current_player
        opponent = 3 - current_player
        
        tensor = torch.zeros(3, self.board_size, self.board_size)
        board = torch.tensor(state.board)
        
        tensor[0] = (board == current_player).float()
        tensor[1] = (board == opponent).float()
        tensor[2] = (board == 0).float()
        
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
        self.network.eval()

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        """执行推理"""
        DEBUG(logger, f"Predicting for state")
        with torch.no_grad():
            state_tensor = self.network._preprocess_state(state).unsqueeze(0).to(self.device)
            policy, value = self.network(state_tensor)
            DEBUG(logger, f"Predicted probabilities: {policy.cpu().numpy().flatten()}, value: {value.item()}")
            return policy.cpu().numpy().flatten(), value.item()
    
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
            
            # 前向传播
            predicted_policies, predicted_values = self.network(states)
            
            # 计算损失
            policy_loss = F.cross_entropy(predicted_policies, target_policies)
            value_loss = F.mse_loss(predicted_values.squeeze(), target_values)
            total_loss = policy_loss/0.9 + value_loss/3.2
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            return policy_loss.item(), value_loss.item()
        finally:
            self.network.eval()

app = FastAPI()
network_server: HexNetTrainerPredictor = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(state: StateData):
    try:
        state_obj = State(np.array(state.board), state.current_player)
        action_probs, value = network_server.predict(state_obj)
        return {
            "action_probs": action_probs.tolist(),
            "value": float(value)
        }
    except Exception as e:
        ERROR(logger, f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train(data: TrainingData):
    try:
        batch = []
        for state, probs, reward in zip(data.states, data.action_probs, data.rewards):
            batch.append({
                'state': State(np.array(state.board), state.current_player),
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
        ERROR(logger, f"Training error: {e}")
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
            policy_channels=config.policy_channels
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
    def __init__(self, config: ExitConfig):
        self.base_url = f"http://{config.network_server_host}:{config.network_server_port}"

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        """请求推理结果"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={
                    "board": state.board.tolist(),
                    "current_player": state.current_player
                }
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result["action_probs"]), result["value"]
        except Exception as e:
            ERROR(logger, f"Prediction error: {e}")
            raise

    def train(self, batch: List[dict]) -> Dict[str, float]:
        """请求训练"""
        try:
            data = {
                "states": [
                    {"board": exp["state"].board.tolist(),
                     "current_player": exp["state"].current_player}
                    for exp in batch
                ],
                "action_probs": [exp["action_probs"].tolist() for exp in batch],
                "rewards": [exp["reward"] for exp in batch]
            }
            response = requests.post(f"{self.base_url}/train", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            ERROR(logger, f"Training error: {e}")
            raise

    def save(self, path: str) -> Dict[str, str]:
        """请求保存模型"""
        response = requests.post(f"{self.base_url}/save", params={"path": path})
        response.raise_for_status()
        return response.json()

    def load(self, path: str) -> Dict[str, str]:
        """请求加载模型"""
        response = requests.post(f"{self.base_url}/load", params={"path": path})
        response.raise_for_status()
        return response.json()

class HexNetWrapper:
    """神经网络服务器包装器"""
    def __init__(self, config: ExitConfig, board_size: int):
        self.config = config
        self.board_size = board_size
        self.network_client = None
        self.server_thread = None

    def start(self):
        """启动服务器和客户端"""
        INFO(logger, "Initializing network server...")
        
        # 创建并启动服务器线程
        self.server_thread = threading.Thread(
            target=run_server,
            kwargs={
                "config": self.config, 
                "board_size": self.board_size,
                "host": self.config.network_server_host,
                "port": self.config.network_server_port
            },
            daemon=True
        )
        self.server_thread.start()
        
        # 创建客户端
        self.network_client = NetworkClient(self.config)
        
        # 等待服务器启动
        import time
        time.sleep(2)  # 给服务器一些启动时间
        
        # 尝试连接服务器
        max_retries = 5
        retry_interval = 1.0
        
        for i in range(max_retries):
            try:
                # 尝试发送一个简单的请求来检查服务器是否就绪
                response = requests.get(f"http://{self.config.network_server_host}:{self.config.network_server_port}/health")
                if response.status_code == 200:
                    INFO(logger, "Successfully connected to network server")
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    WARNING(logger, f"Failed to connect to server, retrying in {retry_interval}s...")
                    time.sleep(retry_interval)
                else:
                    ERROR(logger, "Failed to connect to network server after multiple attempts")
                    raise RuntimeError("Network server initialization failed")
        
        # 如果存在预训练模型，则加载
        if self.config.model_path and os.path.exists(self.config.model_path):
            try:
                result = self.network_client.load(self.config.model_path)
                if result["status"] != "success":
                    ERROR(logger, "Failed to load model")
                    raise RuntimeError("Network server initialization failed")
                INFO(logger, "Network server started successfully")
            except Exception as e:
                ERROR(logger, f"Error during server initialization: {e}")
                raise RuntimeError("Network server initialization failed") from e

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        """请求推理结果"""
        return self.network_client.predict(state)

    def train(self, batch: List[dict]) -> Tuple[float, float]:
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