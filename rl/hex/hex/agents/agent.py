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

class Experience:
    """经验"""
    def __init__(
            self, 
            player_id: int,
            state: State, 
            actions: List[Action],
            action_probs: List[float],
            export_probs: List[float],
            action: Action, 
            reward: float):
        self.player_id = player_id
        self.state = state
        self.actions = actions
        self.action_probs = action_probs
        self.export_probs = export_probs
        self.action = action
        self.reward = reward

    def paint(self) -> str:
        """将棋盘状态绘制为 ASCII 字符画
        
        Returns:
            str: ASCII 字符画表示的棋盘，使用:
                ● 表示黑棋(玩家1)
                ○ 表示白棋(玩家2)
                · 表示空位
                X 表示当前动作位置
                (xx%) 表示该位置的动作概率
        """
        return self.state.board.paint(self.action, self.action_probs)

    def __str__(self) -> str:
        return f"Experience(" \
            f"player_id={self.player_id}, " \
            f"state={self.state}, " \
            f"actions={self.actions}, " \
            f"action_probs={self.action_probs}, " \
            f"export_probs={self.export_probs}, " \
            f"action={self.action}, " \
            f"reward={self.reward})"
            
class ReplayMemory:
    """经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = None

    def append(self, experience: Experience):
        """添加经验"""
        self._new_memory()
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样经验"""
        return random.sample(list(self.memory), batch_size)
    
    def extend(self, experiences: List[Experience]):
        """添加多个经验"""
        self._new_memory()
        self.memory.extend(experiences)
    
    def __len__(self) -> int:
        return len(self.memory) if self.memory else 0
    
    def pop(self):
        """移除指定位置的经验"""
        return self.memory.pop() if self.memory else None

    def thanos(self):
        """随机保留一半的经验"""
        if self.memory:
            # 计算要保留的数量
            keep_size = len(self.memory) // 3
            # 随机选择要保留的经验
            keep_indices = random.sample(range(len(self.memory)), keep_size)
            # 创建新的deque并保留选中的经验
            new_memory = deque(maxlen=self.capacity)
            for i in sorted(keep_indices):
                new_memory.append(self.memory[i])
            self.memory = new_memory

    def clear(self):
        """清空经验回放缓冲区"""
        if self.memory:
            self.memory.clear()
    
    def __getitem__(self, index) -> Any:
        """支持索引和切片访问"""
        if isinstance(index, slice):
            return list(self.memory)[index]
        return self.memory[index]

    def _new_memory(self):
        """创建新的经验回放缓冲区"""
        if self.memory is None:
            self.memory = deque(maxlen=self.capacity)

class GameResult:
    """游戏结果"""
    def __init__(
            self, 
            agent1_id: int, 
            agent2_id: int, 
            first_agent_id: int,
            winner_id: int, 
            experiences: List[Experience],
            moves_count: int,
        ):
        self.agent1_id :int = agent1_id
        self.agent2_id :int = agent2_id
        self.first_agent_id :int = first_agent_id
        self.winner_id :int = winner_id
        self.experiences :List[Experience] = experiences
        self.moves_count = moves_count

    def has_winner(self) -> bool:
        """是否存在获胜者"""
        return self.winner_id is not None
    
    def get_winner(self) -> int:
        """获取获胜者"""
        return self.winner_id

    def __str__(self) -> str:
        return f"GameResult(" \
            f"agent1_id={self.agent1_id}, " \
            f"agent2_id={self.agent2_id}, " \
            f"winner_id={self.winner_id}, " \
            f"experiences={len(self.experiences)}, " \
            f"moves_count={self.moves_count})"
    
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
        self.memory = ReplayMemory(memory_size)
        self.experience = ReplayMemory(memory_size)
        self.memory_size = memory_size

    def choose_action(self, board: Board, cold_start: bool) -> Action:
        """选择动作"""
        if self.policy is None:
            raise ValueError("Policy is not initialized")
            
        state = board.get_state(self.player_id)

        assert state.current_player == self.player_id
        
        action = self.policy.get_action(board, state, cold_start)
        self.current_episode.add_step(
            board=board, 
            state=state, 
            action=action)
        return action
    
    def reward(self, r: float, board: Optional[Board] = None) -> List[Experience]:
        """接收奖励并更新值函数"""
        self.current_episode.set_reward(r)
        experiences = self._store_episode(self.current_episode, r)
        self.memory.extend(experiences)
        if self.estimator and board:
            self.estimator.update(self.current_episode, board)
        self.current_episode = Episode(self.player_id)
        return experiences
    
    def _store_episode(self, episode: Episode, reward: float):
        """存储一局游戏的经历"""
        experiences = []
        for state, actions, probs, export_probs, action in zip(episode.states, episode.actions, episode.probs, episode.export_probs, episode.chosen_actions):
            assert state.current_player == self.player_id
            experience = Experience(
                player_id=self.player_id,
                state=state,
                actions=actions,
                action_probs=probs,
                export_probs=export_probs,
                action=action,
                reward=reward
            )
            experiences.append(experience)
        return experiences
    
    def evaluate(self, experiment: Any, num_games: int = 100) -> float:
        """评估智能体的性能
        
        Args:
            experiment: 游戏实验环境
            num_games: 评估游戏局数
            
        Returns:
            float: 胜率
        """
        wins = 0
        for game in range(num_games):
            experiment.board.reset()
            # 与随机智能体对弈
            opponent = create_random_agent(player_id=3-self.player_id)
            experiment.set_agents(self, opponent)
            
            game_result = experiment.play_game()
            if game_result.winner and game_result.winner.player_id == self.player_id:
                wins += 1
            
        win_rate = wins / num_games
        return win_rate

    def _equal(self, other: Agent) -> bool:
        """判断两个智能体是否相等"""
        return self.player_id == other.player_id

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

