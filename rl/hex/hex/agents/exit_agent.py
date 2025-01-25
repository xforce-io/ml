from __future__ import annotations
import multiprocessing
import sys
import time
from hex.log import ERROR, INFO, WARNING
from hex.agents.agent import Agent, create_random_agent
from hex.config import ExitConfig, ExperimentConfig, MCTSConfig
from hex.hex import Action, Board, State
from hex.agents.mcts_agent import MCTSPolicy
from hex.experiment import ExperimentRunner, HexGameExperiment, GameResult
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
from typing import Tuple, Optional, List
import os
import matplotlib.pyplot as plt
from multiprocessing import Manager

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

class ExitAgent(Agent):
    """Expert Iteration智能体"""
    def __init__(self, 
                 config: ExitConfig,
                 board_size: int,
                 num_cores: int,
                 player_id: int = 1,
                 name: str = "ExIt"):
        self.expert = MCTSPolicy(config.mcts_config, board_size)

        super().__init__(
            self.expert,
            None,
            player_id,
            name,
            memory_size=config.memory_size  # 传递 memory_size 给基类
        )
        
        self.config = config
        self.board_size = board_size
        self.num_cores = num_cores
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化神经网络和优化器
        self.network = HexNet(
            board_size=board_size,
            num_channels=config.num_channels,
            policy_channels=config.policy_channels
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 初始化其他参数
        self.temperature = config.temperature
        self.experiment = HexGameExperiment(board_size=board_size)
        
        # 设置use_network并尝试加载模型
        self.use_network = config.use_network
        if self.use_network and config.model_path:
            self.load_model(config.model_path)
            INFO(logger, f"Loaded model from {config.model_path}")
        
    def choose_action(self, board: Board) -> Action:
        """选择动作，结合神经网络的预测结果"""
        state = board.get_state()
        valid_moves = board.get_valid_moves()
        
        if self.use_network:
            # 使用神经网络进行预测
            self.network.eval()
            with torch.no_grad():
                state_tensor = self.network._preprocess_state(state).unsqueeze(0).to(self.device)
                policy, value = self.network(state_tensor)
                
                # Convert policy tensor to numpy array
                policy = policy.cpu().numpy().flatten()
                
                # Store experience with numpy array
                self._store_experience(state, valid_moves, policy)
                
                # Filter policy for valid moves only
                valid_policy = np.array([policy[move.x * self.board_size + move.y] for move in valid_moves])
                
                # Normalize the valid moves policy
                valid_policy = valid_policy / valid_policy.sum()
                
                # 根据温度参数选择动作
                if self.temperature > 0:
                    valid_policy = np.power(valid_policy, 1/self.temperature)
                    valid_policy = valid_policy / valid_policy.sum()
                    action_idx = np.random.choice(len(valid_moves), p=valid_policy)
                else:
                    action_idx = np.argmax(valid_policy)
                
                chosen_action = valid_moves[action_idx]
                
                # 记录到当前回合
                self.current_episode.add_step(state, chosen_action)
                return chosen_action
        
        # 如果不使用神经网络，使用MCTS专家
        # 创建专家的完整概率分布
        expert_probs = np.zeros(self.board_size * self.board_size)
        action = self.expert.get_action(board, state)
        idx = action.x * self.board_size + action.y
        expert_probs[idx] = 1.0
        
        self._store_experience(state, valid_moves, expert_probs)
        
        self.current_episode.add_step(state, action)
        return action
    
    def train_iteration(self):
        """执行一次训练迭代"""
        if len(self.experience) < self.config.batch_size:
            return
        
        start_time = time.time()
        
        # Move network to training device (GPU if available)
        self.network = self.network.to(self.device)
        
        # 使用ReplayMemory的sample方法
        batch = self.experience.sample(self.config.batch_size)
        
        # 准备训练数据
        states = torch.stack([self.network._preprocess_state(exp['state']) 
                            for exp in batch]).to(self.device)
        
        # 使用numpy.array()预先合并数组，然后一次性转换为tensor
        target_policies = torch.from_numpy(
            np.array([exp['action_probs'] for exp in batch])
        ).float().to(self.device)
        
        target_values = torch.from_numpy(
            np.array([exp['reward'] for exp in batch])
        ).float().to(self.device)
        
        # 前向传播
        self.network.train()
        predicted_policies, predicted_values = self.network(states)  
        # predicted_policies shape: [batch_size, board_size * board_size]
        # predicted_values shape: [batch_size, 1]
        
        # 计算损失
        policy_loss = F.cross_entropy(predicted_policies, target_policies)  # shape: scalar
        value_loss = F.mse_loss(predicted_values.squeeze(), target_values)  # shape: scalar
        total_loss = policy_loss + value_loss  # shape: scalar
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        INFO(logger, f"Training iteration completed - "
                        f"Policy Loss: {policy_loss.item():.4f}, "
                        f"Value Loss: {value_loss.item():.4f}, "
                        f"costSec: {time.time() - start_time:.2f}s")
        
        # Optionally move network back to CPU after training
        if self.device.type == "cuda":
            self.network = self.network.cpu()
    
    def save_model(self, path: str):
        """保存模型"""
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'board_size': self.board_size
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        if not os.path.exists(path):
            WARNING(logger, f"Model file {path} does not exist")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            INFO(logger, f"Successfully loaded model from {path}")
            return True
        except Exception as e:
            ERROR(logger, f"Failed to load model from {path}: {str(e)}")
            return False

    def _process_game_result(self, game_result: GameResult):
        """处理游戏结果，使用MCTS专家对每个状态进行评估"""
        if game_result.winner is None:
            reward = 0.0
        else:
            reward = 1.0 if game_result.winner.player_id == game_result.agent1.player_id else -1.0

        experiences = []
        memory = game_result.agent1.memory
        
        # 创建一个新的棋盘用于MCTS搜索
        board = Board(self.board_size)
        
        for experience in memory:
            state = experience['state']
            # 将棋盘设置为当前状态
            board.set_state(state)
            
            # 使用MCTS专家进行搜索，获取动作概率分布
            self.expert.search(board)  # 运行MCTS搜索
            action_probs = self.expert.get_action_probs(board)  # 获取MCTS访问计数的概率分布
            
            # 更新经验数据，加入MCTS的评估结果
            experience['action_probs'] = action_probs
            experience['reward'] = reward
            experiences.append(experience)
            
            # 重置MCTS搜索树，准备评估下一个状态
            self.expert.reset()

        return experiences

    def train(self):
        """执行完整的训练过程"""
        INFO(logger, f"Starting training with {self.config.num_iterations} iterations")
        INFO(logger, f"Each iteration will play {self.config.self_play_games} self-play games")

        def agentCreator(player_id: int):
            # Create agent with CPU device to avoid CUDA tensor sharing issues
            agent = ExitAgent(
                config=self.config,
                board_size=self.board_size,
                num_cores=self.num_cores,
                player_id=player_id,
                name=f"{self.name}_player_{player_id}"
            )
            # Force CPU device for multiprocessing
            agent.device = torch.device("cpu")
            agent.network = agent.network.to(agent.device)
            return agent

        games_per_process = self.config.self_play_games // self.num_cores
        experiment_runner = ExperimentRunner(
            total_rounds=self.config.self_play_games,
            statistics_rounds=self.config.self_play_games,
            num_cores=self.num_cores
        )

        for iteration in range(self.config.num_iterations):
            # 训练完一次迭代后，开始使用神经网络
            if iteration > 0:  # 第一次迭代后开始使用神经网络
                self.use_network = True
            
            # 运行实验并获取结果
            game_results = experiment_runner.run_experiment_in_parallel(
                lambda: HexGameExperiment(self.board_size),
                lambda: agentCreator(1),
                lambda: agentCreator(2),
                games_per_process,
            )

            # 处理所有游戏结果
            for game_result in game_results:
                experiences = self._process_game_result(game_result)
                for exp in experiences:
                    if len(self.experience) >= self.config.memory_size:
                        self.experience.pop()
                    self.experience.append(exp)

            # 训练网络
            self.train_iteration()
            
            # 降低温度参数（逐渐减少探索）
            self.temperature = max(0.1, self.temperature * 0.95)

            save_path = os.path.join(self.config.model_dir, f"exit_agent_iter_{iteration + 1}.pth")
            self.save_model(save_path)
            INFO(logger, f"Model saved to {save_path}")

    def evaluate(self, experiment: HexGameExperiment, num_games: int = 100) -> float:
        """评估智能体的性能，重写基类方法以处理特殊状态"""
        # 保存原始状态
        original_use_network = self.use_network
        original_temperature = self.temperature
        
        # 设置评估状态
        self.use_network = True
        self.temperature = 0.1
        
        # Ensure network is on the correct device
        self.network = self.network.to(self.device)
        
        try:
            # 调用基类的评估方法
            return super().evaluate(experiment, num_games)
        finally:
            # 恢复原始状态
            self.use_network = original_use_network
            self.temperature = original_temperature
            # Optionally move network back to CPU if needed
            if self.device.type == "cuda":
                self.network = self.network.cpu()

    def _store_experience(self, state: State, valid_moves: List[Action], probs: np.ndarray):
        """存储经验到回放缓冲区
        
        Args:
            state: 当前状态
            valid_moves: 合法动作列表
            probs: 动作概率（可能只包含合法动作的概率）
        """
        # 创建一个完整的概率分布向量（所有位置）
        full_probs = np.zeros(self.board_size * self.board_size)
        
        # 如果输入的probs是针对所有位置的
        if len(probs) == self.board_size * self.board_size:
            full_probs = probs
        # 如果输入的probs只包含合法动作的概率
        else:
            # 将合法动作的概率映射到对应位置
            for action, prob in zip(valid_moves, probs):
                idx = action.x * self.board_size + action.y
                full_probs[idx] = prob
        
        # 确保概率和为1
        if full_probs.sum() > 0:
            full_probs = full_probs / full_probs.sum()
        
        experience = {
            'state': state,
            'actions': valid_moves,
            'action_probs': full_probs  # 现在总是存储完整的概率分布
        }
        self.memory.append(experience)

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
            num_cores=exp_config.num_cores
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

    exp_config = ExperimentConfig(num_cores=200)

    exit_agent = create_exit_agent(board_size=5, player_id=1, exp_config=exp_config)
    opponent = create_random_agent(player_id=2)

    experiment = HexGameExperiment(board_size=5)
    experiment.set_agents(exit_agent, opponent)
    
    # 记录每次评估的胜率
    win_rates = []
    iterations = []
    
    for i in range(10):
        # 评估并记录胜率
        win_rate = exit_agent.evaluate(experiment, 300)
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