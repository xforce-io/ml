from __future__ import annotations
import multiprocessing
import random
import sys
import time
import traceback
from hex.log import ERROR, INFO, DEBUG
from hex.agents.agent import Agent, Experience, create_random_agent
from hex.config import DEBUG_STATE, ExitConfig, ExperimentConfig
from hex.hex import Action, Board, State
from hex.agents.mcts_agent import MCTSPolicy, StatePredictor
from hex.experiment import ExperimentRunner, HexGameExperiment, GameResult
import numpy as np
import logging
from typing import Any, Tuple, Optional, List
import os
import matplotlib.pyplot as plt
from hex.agents.network_server import HexNetWrapper, NetworkClient

logger = logging.getLogger(__name__)

class RemotePredictor(StatePredictor):
    """远程预测器"""
    def __init__(self, client: NetworkClient):
        self.client = client

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        return self.client.predict(state)

class ExitAgent(Agent):
    """Expert Iteration智能体"""
    def __init__(self, 
                 exp_config: ExperimentConfig,
                 config: ExitConfig,
                 board_size: int,
                 num_cores: int,
                 player_id: int = 1,
                 name: str = "ExIt",
                 use_network: bool = True,
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
        self.exp_config = exp_config
        self.board_size = board_size
        self.num_cores = num_cores
        self.use_network = config.use_network and use_network
        
        if hex_net_wrapper is None:
            self.hex_net_wrapper = HexNetWrapper(config, board_size)
            self.hex_net_wrapper.start()
        else:
            self.hex_net_wrapper = hex_net_wrapper

    def clone(
            self, 
            player_id: int, 
            name: Optional[str] = None,
            use_network: bool = True) -> ExitAgent:
        """创建当前智能体的副本"""
        new_agent = ExitAgent(
            exp_config=self.exp_config,
            config=self.config,
            board_size=self.board_size,
            num_cores=self.num_cores,
            player_id=player_id,
            name=name or f"{self.name}_player_{player_id}",
            use_network=use_network,
            hex_net_wrapper=self.hex_net_wrapper.clone()
        )

        self.hex_net_wrapper.start_client()

        # 创建MCTS专家，使用主智能体的网络客户端
        new_agent.expert = MCTSPolicy(
            self.config.mcts_config,
            self.board_size,
            num_threads=1,
            player_id=player_id,
            state_predictor=RemotePredictor(self.hex_net_wrapper.network_client)
        )
        
        return new_agent

    def choose_action(self, board: Board, cold_start: bool) -> Action:
        """选择动作，结合神经网络的预测结果和MCTS搜索"""
        state = board.get_state(self.player_id)
        probs = self.expert.search(board, self.config.mcts_config.simulations, not cold_start and self.use_network)
        action = self.expert.select_action(board, probs)
        self.expert.reset()  # 重置搜索树

        expert_probs = np.zeros(self.board_size * self.board_size)
        idx = action.x * self.board_size + action.y
        expert_probs[idx] = 1.0
        
        self.current_episode.add_step(
            board=board, 
            state=state, 
            action=action, 
            probs=probs,
            export_probs=expert_probs)
        return action
    
    def train_step(self):
        """执行一次训练迭代"""
        if len(self.experience) < self.config.batch_size:
            return
        
        try:
            start_time = time.time()
            batch = self.experience.sample(self.config.batch_size)
            
            # 请求训练
            result = self.hex_net_wrapper.train(batch)
            
            INFO(logger, f"Training iteration completed - "
                      f"Policy Loss: {result[0]:.4f}, "
                      f"Value Loss: {result[1]:.4f}, "
                      f"costSec: {time.time() - start_time:.2f}s")
        except Exception as e:
            ERROR(logger, f"Training error: {e} traceback: {traceback.format_exc()}")

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

    def _process_game_result(self, game_result: GameResult) -> List[Experience]:
        return game_result.experiences

    def _create_game_experiment(self) -> HexGameExperiment:
        """创建游戏实验实例"""
        return HexGameExperiment(self.board_size)
    
    def _create_agent1(self) -> ExitAgent:
        """创建玩家1的智能体"""
        return self.clone(1)
    
    def _create_agent2(self) -> ExitAgent:
        """创建玩家2的智能体"""
        return self.clone(2, use_network=False)

    def _create_random_agent(self) -> ExitAgent:
        """创建随机对手的智能体"""
        return create_random_agent(player_id=2)

    def train_epoch(self, cold_start: bool):
        """执行完整的训练过程"""
        INFO(logger, f"Starting epoch training with {self.config.num_steps_per_epoch} steps cold_start[{cold_start}]")
        
        experiment_runner = ExperimentRunner(
            statistics_rounds=100,
            num_cores=self.num_cores
        )
        
        self.experience.thanos()
        num_experiences_needed = self.config.batch_size * self.config.num_steps_per_epoch
        while len(self.experience) < num_experiences_needed:
            self.accumulate_experience(experiment_runner, cold_start)
            INFO(logger, f"Accumulated experience: {len(self.experience)/num_experiences_needed}")
        
        for iteration in range(self.config.num_steps_per_epoch):
            self.train_step()

            if (iteration + 1) % self.config.save_interval == 0:
                model_path = os.path.join(self.config.model_dir, f"exit_agent_{iteration+1}.pth")
                self.save_model(model_path)
            
    def accumulate_experience(self, experiment_runner: ExperimentRunner, cold_start: bool):
        game_results = experiment_runner.run_experiments(
            gameExperimentCreator=self._create_game_experiment,
            agent1Creator=self._create_agent1,
            agent2Creator=self._create_agent2,
            num_games=self.exp_config.num_cores * 2,
            cold_start=cold_start,
            parallel=self.config.parallel_self_play
        )
        
        # 处理游戏结果，收集训练数据
        for game_result in game_results:
            experiences = self._process_game_result(game_result)
            for exp in experiences:
                if len(self.experience) >= self.config.memory_size:
                    self.experience.pop()
                self.experience.append(exp)

        if DEBUG_STATE:
            num_samples = min(10, len(game_results))
            for game_result in game_results[-num_samples:]:
                for exp in game_result.experiences:
                    print(exp.paint())
                    print("-" * 100)
                print(f"winner: {game_result.winner_id} first_agent_id: {game_result.first_agent_id}")

    def evaluate(self, experiment: Any, num_games: int = 100) -> float:
        """评估智能体的性能"""
        assert num_games % self.num_cores == 0, "num_games must be divisible by num_cores"
        
        # 保存原始状态
        original_use_network = self.use_network
        
        try:
            # 设置评估状态
            self.use_network = True
            
            experiment_runner = ExperimentRunner(
                statistics_rounds=num_games,
                num_cores=self.num_cores
            )
            
            # 运行并行评估
            game_results = experiment_runner.run_experiments(
                gameExperimentCreator=self._create_game_experiment,
                agent1Creator=self._create_agent1,
                agent2Creator=self._create_random_agent,
                num_games=num_games,
                parallel=self.config.parallel_eval
            )
            
            if DEBUG_STATE:
                num_samples = min(10, len(game_results))
                for game_result in game_results[-num_samples:]:
                    for exp in game_result.experiences:
                        print(exp.paint())
                        print("-" * 100)
                    print(f"winner: {game_result.winner_id} first_agent_id: {game_result.first_agent_id}")

            # 统计胜率
            wins = sum(1 for result in game_results 
                      if result.has_winner() and result.get_winner() == 1)
            return wins / len(game_results)
            
        finally:
            # 恢复原始状态
            self.use_network = original_use_network

def create_exit_agent(
        board_size: int, 
        player_id: int, 
        exp_config: Optional[ExperimentConfig] = None) -> ExitAgent:
    """创建并训练Expert Iteration智能体"""
    
    if exp_config is None:
        exp_config = ExperimentConfig(num_cores=1)
    
    # 检查是否存在预训练模型
    model_path = os.path.join(exp_config.model_dir, "exit_agent_final.pth")
    use_network = os.path.exists(model_path)
    
    config = ExitConfig()

    INFO(logger, f"Exit agent config: {config.json()}")
    
    # 创建实验环境和智能体
    agent = ExitAgent(
            exp_config=exp_config,
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
        agent.train_epoch(cold_start=True)
    else:
        INFO(logger, f"Using pre-trained model from {model_path}")
        pass
    
    return agent

if __name__ == "__main__":
    if sys.platform == 'darwin' or sys.platform == 'linux':
        multiprocessing.set_start_method('spawn')

    exp_config = ExperimentConfig(num_cores=10)
    exit_agent = create_exit_agent(
        board_size=exp_config.board_size, 
        player_id=1, 
        exp_config=exp_config)

    win_rates = []
    iterations = []
    exp_no = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
    for i in range(exit_agent.config.num_epochs):
        win_rate = exit_agent.evaluate(None, exp_config.num_games_to_evaluate)
        INFO(logger, f"Evaluation {exp_no} completed 0. Win rate: {win_rate:.2%}")

        win_rate = exit_agent.evaluate(None, exp_config.num_games_to_evaluate)
        INFO(logger, f"Evaluation {exp_no} completed 1. Win rate: {win_rate:.2%}")

        win_rates.append(win_rate)
        iterations.append(i)
        
        # 训练
        exit_agent.train_epoch(cold_start=False)
    
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