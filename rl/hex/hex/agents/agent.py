from __future__ import annotations
from typing import Any, Optional, Dict, List
from hex.log import INFO
import logging
from collections import deque
import random
from hex.config import DynaQConfig
from hex.hex import Action, Board, State
from hex.rl_basic import Episode, LearningConfig, Policy, RLAlgorithm, RandomPolicy, UCBPolicy, ValueEstimator
import numpy as np

logger = logging.getLogger(__name__)

class ReplayMemory:
    """经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def append(self, experience: Dict[str, Any]):
        """添加经验"""
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """随机采样经验"""
        return random.sample(list(self.memory), batch_size)
    
    def extend(self, experiences: List[Dict[str, Any]]):
        """添加多个经验"""
        self.memory.extend(experiences)
    
    def __len__(self) -> int:
        return len(self.memory)
    
    def pop(self):
        """移除指定位置的经验"""
        return self.memory.pop()
    
    def __getitem__(self, index) -> Any:
        """支持索引和切片访问"""
        if isinstance(index, slice):
            return list(self.memory)[index]
        return self.memory[index]

class Agent:
    """智能体"""
    def __init__(
            self, 
            policy: Policy, 
            estimator: Optional[ValueEstimator], 
            player_id: int, 
            name: str = "",
            memory_size: int = 100000):  # 添加 memory_size 参数
        self.policy = policy
        self.estimator = estimator
        self.player_id = player_id
        self.name = name
        self.current_episode = Episode(player_id)
        self.memory = ReplayMemory(memory_size)  # 初始化经验回放缓冲区
    
    def choose_action(self, board: Board) -> Action:
        """选择动作"""
        if self.policy is None:
            raise ValueError("Policy is not initialized")
            
        state = board.get_state()
        action = self.policy.get_action(board, state)
        self.current_episode.add_step(state, action)
        return action
    
    def reward(self, r: float, board: Optional[Board] = None):
        """接收奖励并更新值函数"""
        self.current_episode.set_reward(r)
        if self.estimator and board:
            self.estimator.update(self.current_episode, board)
        self.current_episode = Episode(self.player_id)
    
    def _store_experience(
            self, 
            state: State, 
            actions: List[Action], 
            probs: np.ndarray):
        """存储经验到回放缓冲区"""
        experience = {
            'state': state,
            'actions': actions,
            'action_probs': probs
        }
        self.memory.append(experience)
    
    def evaluate(self, experiment: Any, num_games: int = 100) -> float:
        """评估智能体的性能
        
        Args:
            experiment: 游戏实验环境
            num_games: 评估游戏局数
            
        Returns:
            float: 胜率
        """
        INFO(logger, f"Starting evaluation over {num_games} games")
        wins = 0
        
        for game in range(num_games):
            experiment.board.reset()
            # 与随机智能体对弈
            opponent = create_random_agent(player_id=3-self.player_id)
            experiment.set_agents(self, opponent)
            
            winner, _ = experiment.play_game()
            if winner and winner.player_id == self.player_id:
                wins += 1
            
            if (game + 1) % 10 == 0:
                INFO(logger, f"Completed {game + 1} evaluation games")
        
        win_rate = wins / num_games
        INFO(logger, f"Evaluation completed. Win rate: {win_rate:.2%}")
        return win_rate

def create_random_agent(player_id: int) -> Agent:
    """创建随机智能体"""
    return Agent(RandomPolicy(), None, player_id=player_id, name="Random")

def create_dynaq_agent(config: DynaQConfig, player_id: int) -> Agent:
    """创建DynaQ智能体"""
    learning_config = LearningConfig(
        algorithm_type=config.algorithm_type,
        initial_learning_rate=config.initial_learning_rate,
        final_learning_rate=config.final_learning_rate,
        initial_epsilon=config.initial_epsilon,
        final_epsilon=config.final_epsilon,
        gamma=config.gamma,
        planning_steps=config.planning_steps,
        batch_size=config.batch_size,
        memory_size=config.memory_size
    )
    algorithm = RLAlgorithm(learning_config)
    algorithm.policy = UCBPolicy(algorithm.estimator, c=1.0)
    return Agent(algorithm.policy, algorithm.estimator, player_id=player_id, name=config.name)

