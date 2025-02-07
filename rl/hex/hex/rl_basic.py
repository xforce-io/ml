from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from hex.hex import Action, Board, State
import numpy as np
import random
from typing import List, Optional, Tuple, Dict

@dataclass
class LearningConfig:
    """学习参数配置"""
    algorithm_type: str
    initial_learning_rate: float = 0.2    
    final_learning_rate: float = 0.01     
    initial_epsilon: float = 0.3          # 初始探索率
    final_epsilon: float = 0.05           # 最终探索率
    gamma: float = 0.99                   
    planning_steps: int = 100             
    batch_size: int = 64                  
    memory_size: int = 50000             
    target_update: int = 1000

    def get_learning_rate(self, episode: int, total_episodes: int) -> float:
        """获取当前学习率"""
        decay = episode / total_episodes
        return self.initial_learning_rate * (1 - decay) + self.final_learning_rate * decay

    def get_epsilon(self, episode: int, total_episodes: int) -> float:
        """获取当前探索率"""
        decay = episode / total_episodes
        return self.initial_epsilon * (1 - decay) + self.final_epsilon * decay

class Episode:
    """一局游戏的经历"""
    def __init__(self, player_id: int):
        self.states: List[State] = []
        self.actions: List[List[Action]] = []
        self.probs: List[np.ndarray] = []
        self.chosen_actions: List[Action] = []
        self.player_id = player_id
        self.reward: float = 0
    
    def __len__(self):
        return len(self.states)
    
    def add_step(
            self, 
            board: Board, 
            state: State, 
            action: Action,
            probs: Optional[np.ndarray] = None):
        """添加一步经历"""
        valid_moves = board.get_valid_moves()
        self.states.append(state)
        self.actions.append(valid_moves)
        if probs is not None:
            self.probs.append(self._get_probs(board, valid_moves, probs))
        else:
            probs = np.zeros(board.size * board.size)
            idx = action.x * board.size + action.y
            probs[idx] = 1.0
            self.probs.append(probs)
        self.chosen_actions.append(action)
    
    def set_reward(self, reward: float):
        """设置最终奖励"""
        self.reward = reward

    def _get_probs(
            self, 
            board: Board, 
            actions: List[Action], 
            probs: np.ndarray) -> np.ndarray:
        # 创建一个完整的概率分布向量（所有位置）
        full_probs = np.zeros(board.size * board.size)
        
        # 如果输入的probs是针对所有位置的
        assert len(probs) == board.size * board.size, "概率分布长度不匹配"
        for action in actions:
            idx = action.x * board.size + action.y
            full_probs[idx] = probs[idx]
        
        assert full_probs.sum() > 0, "概率分布和为0"
        return full_probs / full_probs.sum()

class ValueEstimator:
    """值函数估计器基类"""
    def __init__(self, config: LearningConfig):
        self.learning_rate = config.initial_learning_rate
        self.gamma = config.gamma
        self.q_table: Dict[Tuple[State, Action], float] = defaultdict(float)
    
    def get_q_value(self, state: State, action: Action) -> float:
        """获Q值"""
        return self.q_table[(state, action)]

class QLearningEstimator(ValueEstimator):
    """Q-learning值函数估计器"""
    def update(self, episode: Episode, board: Board):
        for i in range(len(episode.states) - 1):
            state = episode.states[i]
            action = episode.chosen_actions[i]
            next_state = episode.states[i + 1]
            reward = episode.reward
            self._update_q_value(state, action, reward, next_state, board)

class SarsaEstimator(ValueEstimator):
    """SARSA值函数估计器"""
    def update(self, episode: Episode, board: Board):
        for i in range(len(episode.states) - 1):
            state = episode.states[i]
            action = episode.chosen_actions[i]
            next_state = episode.states[i + 1]
            next_action = episode.chosen_actions[i + 1]
            reward = episode.reward
            
            # SARSA更新
            next_q = self.get_q_value(next_state, next_action)
            current_q = self.get_q_value(state, action)
            self.q_table[(state, action)] = current_q + self.learning_rate * (
                reward + self.gamma * next_q - current_q
            )

class MonteCarloEstimator(ValueEstimator):
    """Monte Carlo值函数估计器"""
    def __init__(self, config: LearningConfig):
        super().__init__(config)
        self.visit_counts: Dict[Tuple[State, Action], int] = defaultdict(int)
    
    def update(self, episode: Episode, board: Board):
        for state, action in zip(episode.states, episode.chosen_actions):
            key = (state, action)
            self.visit_counts[key] += 1
            count = self.visit_counts[key]
            
            # 增量更新平均值
            old_q = self.q_table[key]
            self.q_table[key] = old_q + (episode.reward - old_q) / count

class Policy:
    """策略基类"""
    def get_action(self, board: Board, state: State) -> Action:
        raise NotImplementedError

class RandomPolicy(Policy):
    """随机策略"""
    def get_action(self, board: Board, state: State) -> Action:
        return random.choice(board.get_valid_moves())

class GreedyPolicy(Policy):
    """贪婪策略"""
    def __init__(self, estimator: ValueEstimator, epsilon: float = 0.1):
        self.estimator = estimator
        self.epsilon = epsilon
    
    def get_action(self, board: Board, state: State) -> Action:
        if random.random() < self.epsilon:
            return random.choice(board.get_valid_moves())
        
        # 获取当前下所有合法动作的Q值
        valid_moves = board.get_valid_moves()
        q_values = [self.estimator.get_q_value(state, action) for action in valid_moves]
        
        # 选择Q值
        max_q = max(q_values)
        max_indices = [i for i, q in enumerate(q_values) if q == max_q]
        chosen_idx = random.choice(max_indices)
        
        return valid_moves[chosen_idx]

class UCBPolicy(Policy):
    """UCB探索策略"""
    def __init__(self, estimator: ValueEstimator, c: float = 1.0):
        self.estimator = estimator
        self.c = c
        self.visit_counts = defaultdict(int)
        
    def get_action(self, board: Board, state: State) -> Action:
        valid_moves = board.get_valid_moves()
        total_visits = sum(self.visit_counts[(state, a)] for a in valid_moves)
        
        def ucb_value(action: Action) -> float:
            q_value = self.estimator.get_q_value(state, action)
            visits = self.visit_counts[(state, action)]
            exploration = self.c * np.sqrt(np.log(total_visits + 1) / (visits + 1))
            return q_value + exploration
        
        return max(valid_moves, key=ucb_value)

class RLAlgorithm:
    """强化学习算法实现"""
    def __init__(self, config: LearningConfig):
        self.config = config
        self.estimator = self._create_estimator()
        self.policy = self._create_policy()
        
    def _create_estimator(self) -> ValueEstimator:
        """根据配置创建值函数估计器"""
        if self.config.algorithm_type == 'Q-learning':
            return QLearningEstimator(self.config)
        elif self.config.algorithm_type == 'SARSA':
            return SarsaEstimator(self.config)
        elif self.config.algorithm_type == 'Monte-Carlo':
            return MonteCarloEstimator(self.config)
        elif self.config.algorithm_type == 'DynaQ':
            return QLearningEstimator(self.config)  # DynaQ 使用 Q-learning 估计器
        else:
            raise ValueError(f"Unknown algorithm type: {self.config.algorithm_type}")
    
    def _create_policy(self) -> Policy:
        """创建策略"""
        if self.config.algorithm_type == 'DynaQ':
            # DynaQ 使用 UCB 策略
            return UCBPolicy(self.estimator, c=1.0)
        else:
            # 其他算法使用 ε-贪婪策略
            return GreedyPolicy(self.estimator, epsilon=self.config.initial_epsilon)
    
    def update(self, state: State, action: Action, reward: float, 
               next_state: State, board: Board):
        """更新算法"""
        if self.config.algorithm_type == 'DynaQ':
            self._update_dynaq(state, action, reward, next_state, board)
        else:
            self.estimator._update_q_value(state, action, reward, next_state, board)
    
    def _update_dynaq(self, state: State, action: Action, reward: float, 
                      next_state: State, board: Board):
        """DynaQ 算法更新"""
        # 实际经验更新
        self.estimator._update_q_value(state, action, reward, next_state, board)
        
        # 规划更新
        for _ in range(self.config.planning_steps):
            # 从经验中随机采样
            sampled_state = self._sample_state(board)
            sampled_action = random.choice(board.get_valid_moves())
            
            # 使用模型预测
            next_board = board.copy()
            next_board.make_move(sampled_action)
            predicted_next_state = next_board.get_state()
            predicted_reward = self._predict_reward(sampled_state, sampled_action)
            
            # 更新Q值
            self.estimator._update_q_value(
                sampled_state, 
                sampled_action,
                predicted_reward,
                predicted_next_state,
                next_board
            )
    
    def _sample_state(self, board: Board) -> State:
        """从经验中采样状态"""
        # 简单实现：随机生成一个合法状态
        new_board = board.copy()
        valid_moves = new_board.get_valid_moves()
        if valid_moves:
            action = random.choice(valid_moves)
            new_board.make_move(action)
        return new_board.get_state()
    
    def _predict_reward(self, state: State, action: Action) -> float:
        """预测奖励"""
        # 简单实现：使用当前Q值作为预测
        return self.estimator.get_q_value(state, action)
    
    def get_action(self, board: Board, state: State) -> Action:
        """获取动作"""
        return self.policy.get_action(board, state)
