from __future__ import annotations
import multiprocessing
import sys
import time
from hex.log import ERROR, INFO, DEBUG
from hex.agents.agent import Agent, create_random_agent
from hex.config import ExitConfig, ExperimentConfig
from hex.hex import Action, Board, State
from hex.agents.mcts_agent import MCTSPolicy, StatePredictor
from hex.experiment import ExperimentRunner, HexGameExperiment, GameResult
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional, List
import os
import matplotlib.pyplot as plt
from hex.agents.network_server import HexNetWrapper, NetworkClient

logger = logging.getLogger(__name__)

class HexNet(nn.Module):
    """Hex游戏的神经网络模型"""
    def __init__(self, board_size: int, num_channels: int, policy_channels: int = 32):
        super().__init__()
        self.board_size = board_size
        
        # 共享特征提取层
        self.conv1 = nn.Conv2d(3, num_channels, 3, padding=1)  # 3个通道：当前玩家棋子、对手棋子、空位
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
    
    def _preprocess_state(self, state: State) -> torch.Tensor:
        """将状态转换为神经网络输入格式"""
        # 创建3个通道：当前玩家棋子、对手棋子、空位
        current_player = state.current_player
        opponent = 3 - current_player
        
        tensor = torch.zeros(3, self.board_size, self.board_size)  # shape: [3, board_size, board_size]
        board = torch.tensor(state.board)  # shape: [board_size, board_size]
        
        tensor[0] = (board == current_player).float()  # shape: [board_size, board_size]
        tensor[1] = (board == opponent).float()        # shape: [board_size, board_size]
        tensor[2] = (board == 0).float()              # shape: [board_size, board_size]
        
        return tensor  # shape: [3, board_size, board_size]
    
    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        """预测策略和价值"""
        state_tensor = self._preprocess_state(state).unsqueeze(0).to(self.device)
        policy, value = self(state_tensor)
        return policy[0].cpu().numpy().flatten(), value[0].item()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # x shape: [batch_size, 3, board_size, board_size]
        
        # 特征提取
        x = F.relu(self.conv1(x))  # shape: [batch_size, num_channels, board_size, board_size]
        x = F.relu(self.conv2(x))  # shape: [batch_size, num_channels, board_size, board_size]
        x = F.relu(self.conv3(x))  # shape: [batch_size, num_channels, board_size, board_size]
        
        # 策略头
        policy = F.relu(self.policy_conv(x))  # shape: [batch_size, 32, board_size, board_size]
        policy = policy.view(-1, 32 * self.board_size * self.board_size)  # shape: [batch_size, 32 * board_size * board_size]
        policy = self.policy_fc(policy)  # shape: [batch_size, board_size * board_size]
        policy = F.softmax(policy, dim=1)  # shape: [batch_size, board_size * board_size]
        
        # 价值头
        value = F.relu(self.value_conv(x))  # shape: [batch_size, 32, board_size, board_size]
        value = value.view(-1, 32 * self.board_size * self.board_size)  # shape: [batch_size, 32 * board_size * board_size]
        value = F.relu(self.value_fc1(value))  # shape: [batch_size, 256]
        value = torch.tanh(self.value_fc2(value))  # shape: [batch_size, 1]
        
        return policy, value

class RemotePredictor(StatePredictor):
    """远程预测器"""
    def __init__(self, client: NetworkClient):
        self.client = client

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        return self.client.predict(state)

class ExitAgent(Agent):
    """Expert Iteration智能体"""
    def __init__(self, 
                 config: ExitConfig,
                 board_size: int,
                 num_cores: int,
                 player_id: int = 1,
                 name: str = "ExIt",
                 hex_net_wrapper: Optional[HexNetWrapper] = None):
        super().__init__(
            policy=None,
            estimator=None,
            player_id=player_id,
            name=name,
            memory_size=config.memory_size
        )
        
        self.expert :MCTSPolicy = None
        self.config = config
        self.board_size = board_size
        self.num_cores = num_cores
        self.use_network = config.use_network
        self.temperature = config.temperature
        
        if hex_net_wrapper is None:
            INFO(logger, "Creating network wrapper...")
            self.hex_net_wrapper = HexNetWrapper(config, board_size)
            self.hex_net_wrapper.start()
            INFO(logger, "Network wrapper started")
        else:
            self.hex_net_wrapper = hex_net_wrapper

    def clone(self, player_id: int, name: Optional[str] = None) -> ExitAgent:
        """创建当前智能体的副本"""
        new_agent = ExitAgent(
            config=self.config,
            board_size=self.board_size,
            num_cores=1,
            player_id=player_id,
            name=name or f"{self.name}_player_{player_id}",
            hex_net_wrapper=self.hex_net_wrapper
        )
        
        # 创建MCTS专家，使用主智能体的网络客户端
        new_agent.expert = MCTSPolicy(
            self.config.mcts_config,
            self.board_size,
            num_threads=1,
            player_id=player_id,
            state_predictor=RemotePredictor(self.hex_net_wrapper.network_client)
        )
        
        return new_agent

    def choose_action(self, board: Board) -> Action:
        """选择动作，结合神经网络的预测结果和MCTS搜索"""
        state = board.get_state()
        action = self.expert.search_and_select_action(board)
        expert_probs = np.zeros(self.board_size * self.board_size)
        idx = action.x * self.board_size + action.y
        expert_probs[idx] = 1.0
        
        self.current_episode.add_step(
            board=board, 
            state=state, 
            action=action, 
            probs=expert_probs)
        return action
    
    def train_iteration(self):
        """执行一次训练迭代"""
        if len(self.experience) < self.config.batch_size:
            return
        
        try:
            start_time = time.time()
            batch = self.experience.sample(self.config.batch_size)
            
            # 请求训练
            result = self.hex_net_wrapper.train(batch)
            
            INFO(logger, f"Training iteration completed - "
                        f"Policy Loss: {result['policy_loss']:.4f}, "
                        f"Value Loss: {result['value_loss']:.4f}, "
                        f"Total Loss: {result['total_loss']:.4f}, "
                        f"costSec: {time.time() - start_time:.2f}s")
        except Exception as e:
            ERROR(logger, f"Training error: {e}")

    def save_model(self, path: str):
        """保存模型"""
        try:
            result = self.hex_net_wrapper.save(path)
            if result["status"] != "success":
                ERROR(logger, f"Failed to save model")
                return False
            INFO(logger, f"Model saved to {path}")
            return True
        except Exception as e:
            ERROR(logger, f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str):
        """加载模型"""
        try:
            result = self.hex_net_wrapper.load(path)
            if result["status"] != "success":
                ERROR(logger, f"Failed to load model")
                return False
            INFO(logger, f"Model loaded from {path}")
            return True
        except Exception as e:
            ERROR(logger, f"Error loading model: {e}")
            return False

    def _process_game_result(self, game_result: GameResult) -> List[dict]:
        return game_result.experiences1

    def _create_game_experiment(self) -> HexGameExperiment:
        """创建游戏实验实例"""
        return HexGameExperiment(self.board_size)
    
    def _create_agent1(self) -> ExitAgent:
        """创建玩家1的智能体"""
        return self.clone(1)
    
    def _create_agent2(self) -> ExitAgent:
        """创建玩家2的智能体"""
        return self.clone(2)

    def train(self):
        """执行完整的训练过程"""
        INFO(logger, f"Starting training with {self.config.num_iterations} iterations")
        
        experiment_runner = ExperimentRunner(
            statistics_rounds=self.config.self_play_games,
            num_cores=self.num_cores
        )
        
        # 训练循环
        for iteration in range(self.config.num_iterations):
            # 自我对弈收集数据
            game_results = experiment_runner.run_experiments(
                gameExperimentCreator=self._create_game_experiment,
                agent1Creator=self._create_agent1,
                agent2Creator=self._create_agent2,
                num_games=self.config.self_play_games,
                parallel=self.config.parallel_self_play
            )
            
            # 处理游戏结果，收集训练数据
            for game_result in game_results:
                experiences = self._process_game_result(game_result)
                for exp in experiences:
                    if len(self.experience) >= self.config.memory_size:
                        self.experience.pop()
                    self.experience.append(exp)
            
            # 训练网络
            self.train_iteration()
            
            # 每隔一定轮次保存模型
            if (iteration + 1) % self.config.save_interval == 0:
                model_path = os.path.join(self.config.model_dir, f"exit_agent_{iteration+1}.pth")
                self.save_model(model_path)
            
            # 降低温度参数
            self.temperature = max(0.1, self.temperature * 0.95)

    def evaluate(self, experiment: HexGameExperiment, num_games: int = 100) -> float:
        """评估智能体的性能"""
        assert num_games % self.num_cores == 0, "num_games must be divisible by num_cores"
        
        # 保存原始状态
        original_use_network = self.use_network
        original_temperature = self.temperature
        
        try:
            # 设置评估状态
            self.use_network = True
            self.temperature = 0.1
            
            experiment_runner = ExperimentRunner(
                statistics_rounds=num_games,
                num_cores=self.num_cores
            )
            
            # 运行并行评估
            game_results = experiment_runner.run_experiments(
                gameExperimentCreator=lambda: experiment,
                agent1Creator=lambda: self.clone(1),  # 评估智能体总是玩家1
                agent2Creator=lambda: create_random_agent(2),  # 随机对手总是玩家2
                num_games=num_games,
                parallel=self.config.parallel_eval
            )
            
            # 统计胜率
            wins = sum(1 for result in game_results 
                      if result.has_winner() and result.get_winner() == 1)
            win_rate = wins / len(game_results)
            
            INFO(logger, f"Evaluation completed. Win rate: {win_rate:.2%}")
            return win_rate
            
        finally:
            # 恢复原始状态
            self.use_network = original_use_network
            self.temperature = original_temperature

def create_exit_agent(
        board_size: int = 5, 
        player_id: int = 1, 
        exp_config: Optional[ExperimentConfig] = None) -> ExitAgent:
    """创建并训练Expert Iteration智能体"""
    
    if exp_config is None:
        exp_config = ExperimentConfig(num_cores=1)
    
    # 检查是否存在预训练模型
    model_path = os.path.join(exp_config.model_dir, "exit_agent_final.pth")
    use_network = os.path.exists(model_path)
    
    config = ExitConfig()
    
    # 创建实验环境和智能体
    agent = ExitAgent(
            config=config, 
            board_size=board_size, 
            player_id=player_id,
            name="ExIt-Agent",
            num_cores=exp_config.num_cores,
            hex_net_wrapper=None
        )
    
    # 如果不使用预训练模型，进行训练
    if not use_network:
        INFO(logger, "No pre-trained model found. Starting training...")
        agent.train()
    else:
        INFO(logger, f"Using pre-trained model from {model_path}")
    
    return agent

if __name__ == "__main__":
    if sys.platform == 'darwin' or sys.platform == 'linux':
        multiprocessing.set_start_method('spawn')

    exp_config = ExperimentConfig(num_cores=5)

    exit_agent = create_exit_agent(board_size=5, player_id=1, exp_config=exp_config)
    opponent = create_random_agent(player_id=2)

    experiment = HexGameExperiment(board_size=5)
    experiment.set_agents(exit_agent, opponent)
    
    # 记录每次评估的胜率
    win_rates = []
    iterations = []
    
    for i in range(10):
        # 评估并记录胜率
        win_rate = exit_agent.evaluate(experiment, 50)
        win_rates.append(win_rate)
        iterations.append(i)
        
        # 训练
        exit_agent.train()
    
    # 绘制胜率变化图
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, win_rates, 'b-', marker='o')
    plt.title('ExIt Agent 训练过程中的胜率变化')
    plt.xlabel('迭代次数')
    plt.ylabel('胜率')
    plt.grid(True)
    plt.ylim(0, 1)
    
    # 保存图表
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig('data/plots/win_rate_curve.png')
    INFO(logger, "Win rate curve saved to data/plots/win_rate_curve.png")