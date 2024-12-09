from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from collections import defaultdict
import random
import logging
import math
import concurrent.futures
import threading
import time
import os
import sys
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class Action:
    """表示一个落子动作"""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

class State:
    """表示棋盘状态"""
    def __init__(self, board: np.ndarray, current_player: int):
        self.board = board.copy()
        self.current_player = current_player
        self.connectivity_features = self._calculate_connectivity()
        self.distance_to_goal = self._calculate_distance_to_goal()
    
    def __hash__(self):
        return hash((self.board.tobytes(), self.current_player))
    
    def __eq__(self, other):
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)
    
    def _calculate_connectivity(self) -> np.ndarray:
        """计算棋盘连通性特征"""
        # 实现连通性计算
        pass
        
    def _calculate_distance_to_goal(self) -> float:
        """计算到目标的最短距离"""
        # 实现距离计算
        pass

class Board:
    """Hex游戏棋盘"""
    def __init__(self, size: int = 5):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        
    def reset(self):
        """重置棋盘"""
        self.board.fill(0)
        self.current_player = 1
    
    def get_state(self) -> State:
        """获取当前状态"""
        return State(self.board, self.current_player)
    
    def is_valid_move(self, action: Action) -> bool:
        """检查动是否合法"""
        return (0 <= action.x < self.size and 
                0 <= action.y < self.size and 
                self.board[action.x, action.y] == 0)
    
    def get_valid_moves(self) -> List[Action]:
        """获取所有合法动作"""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                action = Action(x, y)  # 移除 player 参数
                if self.is_valid_move(action):
                    moves.append(action)
        return moves
    
    def make_move(self, action: Action) -> Tuple[bool, float]:
        """执行一步动作，返回（是否游戏结束，奖励）"""
        if not self.is_valid_move(action):
            return False, -1.0
        
        self.board[action.x, action.y] = self.current_player
        
        # 检查当前玩家是否获胜
        if self.check_win(self.current_player):
            return True, 1.0
        
        # 切换玩家
        self.current_player = 3 - self.current_player
        
        # 检查对手是否已经获胜（在之前的回合）
        if self.check_win(3 - self.current_player):
            return True, -1.0  # 游戏结束，当玩家失败
            
        return False, 0.0  # 游戏继续
    
    def check_win(self, player: int) -> bool:
        """检家是否获"""
        # 使用深度优先搜索检查否连通
        def dfs(x: int, y: int, visited: set) -> bool:
            if player == 1 and y == self.size - 1:  # 玩家1需要连接左右
                return True
            if player == 2 and x == self.size - 1:  # 玩家2需要连接上下
                return True
            
            directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) not in visited and \
                   0 <= new_x < self.size and \
                   0 <= new_y < self.size and \
                   self.board[new_x, new_y] == player:
                    visited.add((new_x, new_y))
                    if dfs(new_x, new_y, visited):
                        return True
            return False
        
        # 检查起始边
        visited = set()
        if player == 1:  # 检查左边
            for x in range(self.size):
                if self.board[x, 0] == player:
                    visited.add((x, 0))
                    if dfs(x, 0, visited):
                        return True
        else:  # 检查上边
            for y in range(self.size):
                if self.board[0, y] == player:
                    visited.add((0, y))
                    if dfs(0, y, visited):
                        return True
        return False
    
    def copy(self) -> 'Board':
        """创建棋盘的深拷贝"""
        new_board = Board.__new__(Board)  # 避免调用 __init__
        new_board.size = self.size
        new_board.board = self.board.copy()  # numpy的copy是高效的
        new_board.current_player = self.current_player
        return new_board

@dataclass
class LearningConfig:
    """学习参数一配置"""
    algorithm_type: str
    initial_learning_rate: float = 0.2    # 更高的初始学习率
    final_learning_rate: float = 0.01     # 最终学习率
    initial_epsilon: float = 0.3          # 更高的初始探索率
    final_epsilon: float = 0.05           # 最终探索率
    gamma: float = 0.99                   # 增加折扣因子
    min_visits: int = 10                  # 增加最小访问次数
    planning_steps: int = 100             # 增加规划步数
    batch_size: int = 64                  # 增加批量大小
    memory_size: int = 50000             # 增加经验池大小
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
        self.actions: List[Action] = []
        self.player_id = player_id
        self.reward: float = 0
    
    def add_step(self, state: State, action: Action):
        """添加一步经历"""
        self.states.append(state)
        self.actions.append(action)
    
    def set_reward(self, reward: float):
        """设置最终奖励"""
        self.reward = reward

class ValueEstimator:
    """值函数估计器基类"""
    def __init__(self, config: LearningConfig):
        self.learning_rate = config.initial_learning_rate
        self.gamma = config.gamma
        self.q_table: Dict[Tuple[State, Action], float] = defaultdict(float)
    
    def get_q_value(self, state: State, action: Action) -> float:
        """获Q值"""
        return self.q_table[(state, action)]
    
    def _update_q_value(self, state: State, action: Action, reward: float, 
                       next_state: State, board: Board):
        """通用的Q值更新逻辑"""
        next_valid_moves = board.get_valid_moves()
        if not next_valid_moves:
            next_max_q = 0
        else:
            next_max_q = max([self.get_q_value(next_state, next_action) 
                            for next_action in next_valid_moves])
        
        current_q = self.get_q_value(state, action)
        self.q_table[(state, action)] = current_q + self.learning_rate * (
            reward + self.gamma * next_max_q - current_q
        )

class QLearningEstimator(ValueEstimator):
    """Q-learning值函数估计器"""
    def update(self, episode: Episode, board: Board):
        for i in range(len(episode.states) - 1):
            state = episode.states[i]
            action = episode.actions[i]
            next_state = episode.states[i + 1]
            reward = episode.reward
            self._update_q_value(state, action, reward, next_state, board)

class SarsaEstimator(ValueEstimator):
    """SARSA值函数估计器"""
    def update(self, episode: Episode, board: Board):
        for i in range(len(episode.states) - 1):
            state = episode.states[i]
            action = episode.actions[i]
            next_state = episode.states[i + 1]
            next_action = episode.actions[i + 1]
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
        for state, action in zip(episode.states, episode.actions):
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

class MCTSNode:
    """MCTS树节点"""
    def __init__(self, state: State, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Action] = None, use_rave: bool = False):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[Action, MCTSNode] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[Action] = []
        
        # RAVE相关属性
        self.use_rave = use_rave
        if use_rave:
            self.rave_visits = 0
            self.rave_value = 0.0
    
    def update(self, reward: float):
        """更新节点统计"""
        self.visits += 1
        self.value += reward
    
    def update_rave(self, reward: float):
        """更新RAVE统计"""
        if self.use_rave:
            self.rave_visits += 1
            self.rave_value += reward
    
    def add_child(self, action: Action, child: 'MCTSNode'):
        """添加子节点"""
        self.children[action] = child
    
    def remove_untried_action(self, action: Action):
        """移除未尝试的动作"""
        self.untried_actions.remove(action)
    
    def is_terminal(self) -> bool:
        """判断是否为终端节点"""
        return len(self.untried_actions) == 0 and len(self.children) == 0
    
    def is_fully_expanded(self) -> bool:
        """判断是否完展"""
        return len(self.untried_actions) == 0
    
    def get_value(self, c: float = 1.414, rave_constant: float = 300) -> float:
        """节点评估"""
        if self.visits == 0:
            return float('inf')
            
        mc_score = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        if not self.use_rave:
            return mc_score + exploration
            
        # RAVE评估
        beta = (rave_constant / (3 * self.visits + rave_constant)) ** 2
        rave_score = self.rave_value / (self.rave_visits + 1e-5)
        return (1 - beta) * (mc_score + exploration) + beta * rave_score
    
    def get_children(self) -> Dict[Action, MCTSNode]:
        """安全地获取children"""
        return dict(self.children)  # 返回副本

class MCTSPolicy(Policy):
    """基于MCTS的策略"""
    def __init__(self, simulations_per_move: int = 3000,
                 c: float = 1.414,
                 rave_constant: float = 300,
                 use_rave: bool = False,
                 selection_strategy: str = 'robust',
                 player_id: int = 1,
                 max_depth: int = 50,
                 base_rollouts_per_leaf: int = 20):  # 基础rollout次数
        self.simulations_per_move = simulations_per_move
        self.c = c
        self.rave_constant = rave_constant
        self.use_rave = use_rave
        self.selection_strategy = selection_strategy
        self.player_id = player_id
        self.max_depth = max_depth
        self.base_rollouts_per_leaf = base_rollouts_per_leaf
    
    def _get_dynamic_rollouts(self, board: Board) -> int:
        """根据剩余空格动态调整rollout次数"""
        empty_spaces = len(board.get_valid_moves())
        total_spaces = board.size * board.size
        
        # 根据剩余空格比例调整rollout次数
        # 游戏后期（空格少）时增加rollout次数
        ratio = 1.0 - (empty_spaces / total_spaces)  # 比例从0到1
        additional_rollouts = int(10 * ratio)  # 最多额外增加10次
        
        return self.base_rollouts_per_leaf + additional_rollouts
    
    def _simulate(self, board: Board) -> float:
        """改进的模拟策略，动态调整rollout次数"""
        rewards = []
        original_board = board.copy()
        rollouts = self._get_dynamic_rollouts(original_board)
        
        # 对同一个叶子节点进行多次rollout
        for _ in range(rollouts):
            board = original_board.copy()
            original_player = board.current_player
            depth = 0
            
            while depth < self.max_depth:
                valid_moves = board.get_valid_moves()
                if not valid_moves:
                    rewards.append(0.0)
                    break
                
                # 动态调整策略
                if depth < self.max_depth // 4:  # 开局阶段
                    action = self._get_early_game_action(board, valid_moves)
                elif depth < self.max_depth * 3 // 4:  # 中局阶段
                    if random.random() < 0.8:
                        action = self._get_heuristic_action(board, valid_moves)
                    else:
                        action = random.choice(valid_moves)
                else:  # 残局阶段
                    action = self._get_endgame_action(board, valid_moves)
                
                game_over, reward = board.make_move(action)
                depth += 1
                
                if game_over:
                    rewards.append(1.0 if board.current_player == original_player else -1.0)
                    break
            
            if depth >= self.max_depth:
                rewards.append(self._evaluate_position(board, original_player))
        
        # 返回所有rollout的平均奖励
        return sum(rewards) / len(rewards)
    
    def get_action(self, board: Board, state: State) -> Action:
        root = MCTSNode(state, use_rave=self.use_rave)
        root.untried_actions = board.get_valid_moves()
        
        # 串行执行模拟
        for _ in range(self.simulations_per_move):
            self._run_simulation(root, board.copy())
        
        return self._select_final_action(root)
    
    def _run_simulation(self, root: MCTSNode, board: Board) -> None:
        """运行单次模拟"""
        node = root
        sim_board = board.copy()
        
        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = self._select_child(node)
            sim_board.make_move(node.action)
        
        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node, sim_board)
        
        # Simulation
        reward = self._simulate(sim_board)
        
        # Backpropagation
        self._backpropagate(node, reward)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """选子节点"""
        return max(node.children.values(), 
                  key=lambda n: n.get_value(self.c, self.rave_constant))
    
    def _expand(self, node: MCTSNode, board: Board) -> MCTSNode:
        """扩展节点"""
        action = node.untried_actions.pop()
        board.make_move(action)
        child_state = board.get_state()
        child = MCTSNode(child_state, parent=node, action=action, 
                        use_rave=self.use_rave)
        child.untried_actions = board.get_valid_moves()
        node.children[action] = child
        return child
    
    def _get_early_game_action(self, board: Board, valid_moves: List[Action]) -> Action:
        """开局策略"""
        # 先选择中心区域
        center = board.size // 2
        center_moves = []
        for move in valid_moves:
            if abs(move.x - center) <= 1 and abs(move.y - center) <= 1:
                center_moves.append(move)
        
        if center_moves:
            return random.choice(center_moves)
        return self._get_heuristic_action(board, valid_moves)
    
    def _get_endgame_action(self, board: Board, valid_moves: List[Action]) -> Action:
        """残局策略"""
        best_score = float('-inf')
        best_moves = []
        
        for move in valid_moves:
            temp_board = board.copy()
            temp_board.make_move(move)
            
            # 评估这步棋后的局面
            score = self._evaluate_position(temp_board, board.current_player)
            
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        
        return random.choice(best_moves)
    
    def _get_heuristic_action(self, board: Board, valid_moves: List[Action]) -> Action:
        """改进的启发式动作选择"""
        move_scores = []
        center = board.size // 2
        
        for move in valid_moves:
            # 计算多个特征
            distance_to_center = abs(move.x - center) + abs(move.y - center)
            connectivity_score = self._evaluate_connectivity(board, move)
            bridge_score = self._evaluate_bridge_potential(board, move)
            blocking_score = self._evaluate_blocking_value(board, move)
            
            # 综合评分
            score = (0.3 * connectivity_score +
                    0.3 * bridge_score +
                    0.3 * blocking_score -
                    0.1 * (distance_to_center / board.size))
            
            move_scores.append((score, move))
        
        # 选择最佳动作
        move_scores.sort(key=lambda x: x[0], reverse=True)
        top_k = max(3, len(move_scores) // 4)  # 动态调整选择范围
        return random.choice([m[1] for m in move_scores[:top_k]])
    
    def _evaluate_connectivity(self, board: Board, move: Action) -> float:
        """评估一个动作的连接价值"""
        # 检查周围8个方向的己方棋子数量
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        connected_pieces = 0
        
        for dx, dy in directions:
            x, y = move.x + dx, move.y + dy
            if (0 <= x < board.size and 0 <= y < board.size and 
                board.board[x, y] == board.current_player):
                connected_pieces += 1
        
        return connected_pieces / len(directions)
    
    def _evaluate_position(self, board: Board, original_player: int) -> float:
        """改进的位置评估函数"""
        # 计算基础分数
        player_pieces = np.sum(board.board == original_player)
        opponent_pieces = np.sum(board.board == (3 - original_player))
        
        # 计算连接度
        player_connectivity = self._calculate_board_connectivity(board, original_player)
        opponent_connectivity = self._calculate_board_connectivity(board, 3 - original_player)
        
        # 计算到目标边的距离
        player_distance = self._calculate_distance_to_goal(board, original_player)
        opponent_distance = self._calculate_distance_to_goal(board, 3 - original_player)
        
        # 计算中心控制
        player_center_control = self._calculate_center_control(board, original_player)
        opponent_center_control = self._calculate_center_control(board, 3 - original_player)
        
        # 计算潜在胜利路径
        player_paths = self._calculate_potential_winning_paths(board, original_player)
        opponent_paths = self._calculate_potential_winning_paths(board, 3 - original_player)
        
        # 调整权重分配
        score = (0.15 * (player_pieces - opponent_pieces) / (board.size * board.size) +
                0.25 * (player_connectivity - opponent_connectivity) +
                0.25 * (opponent_distance - player_distance) +
                0.15 * (player_center_control - opponent_center_control) +
                0.20 * (player_paths - opponent_paths))
        
        return max(min(score, 1.0), -1.0)
    
    def _calculate_distance_to_goal(self, board: Board, player: int) -> float:
        """计算到目标边的最短距离"""
        min_distance = float('inf')
        size = board.size
        
        if player == 1:  # 横向连接
            # 检查每个己方棋子到右边界的距离
            for x in range(size):
                for y in range(size):
                    if board.board[x, y] == player:
                        distance = size - 1 - y
                        min_distance = min(min_distance, distance)
        else:  # 纵向连接
            # 检查每个己方棋子到下边界的距离
            for x in range(size):
                for y in range(size):
                    if board.board[x, y] == player:
                        distance = size - 1 - x
                        min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else size
    
    def _calculate_center_control(self, board: Board, player: int) -> float:
        """计算中心区域控��度"""
        size = board.size
        center = size // 2
        center_score = 0.0
        total_weights = 0.0
        
        for x in range(size):
            for y in range(size):
                if board.board[x, y] == player:
                    # 距离中心越近权重越大
                    weight = 1.0 / (1.0 + abs(x - center) + abs(y - center))
                    center_score += weight
                    total_weights += weight
        
        return center_score / total_weights if total_weights > 0 else 0.0
    
    def _calculate_board_connectivity(self, board: Board, player: int) -> float:
        """计算整个棋盘上某个玩家的连接度"""
        connectivity = 0
        for x in range(board.size):
            for y in range(board.size):
                if board.board[x, y] == player:
                    directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
                    for dx, dy in directions:
                        new_x, new_y = x + dx, y + dy
                        if (0 <= new_x < board.size and 
                            0 <= new_y < board.size and 
                            board.board[new_x, new_y] == player):
                            connectivity += 1
        
        return connectivity / (board.size * board.size * 6)  # 归一化
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """改进的反向传播"""
        path = []
        actions_seen = set()
        
        # 收集路径
        current = node
        while current is not None:
            path.append(current)
            if current.action:
                actions_seen.add(current.action)
            current = current.parent
        
        # 反向传播
        for node in path:
            # 更新节点统计
            node.update(reward if node.state.current_player == self.player_id else -reward)
            
            # RAVE更新
            if node.use_rave and node.parent:
                # 获取父节点children的快照
                children = node.parent.get_children()
                for action, child in children.items():
                    if action in actions_seen:
                        child.update_rave(
                            reward if node.state.current_player == self.player_id else -reward
                        )
            
            reward = -reward

    def _calculate_potential_winning_paths(self, board: Board, player: int) -> float:
        """计算潜在获胜路径数"""
        size = board.size
        paths = 0
        
        if player == 1:  # 横向连接
            for x in range(size):
                empty_count = 0
                player_count = 0
                for y in range(size):
                    if board.board[x, y] == 0:
                        empty_count += 1
                    elif board.board[x, y] == player:
                        player_count += 1
                # 如果这条路径上没有对手的棋子
                if player_count + empty_count == size:
                    paths += (1.0 + 0.5 * player_count)  # 已有棋子越多，分数越高
        else:  # 纵向连接
            for y in range(size):
                empty_count = 0
                player_count = 0
                for x in range(size):
                    if board.board[x, y] == 0:
                        empty_count += 1
                    elif board.board[x, y] == player:
                        player_count += 1
                if player_count + empty_count == size:
                    paths += (1.0 + 0.5 * player_count)
        
        return paths / size  # 归一化

    def _select_final_action(self, root: MCTSNode) -> Action:
        """根据模拟结果选择最终动作"""
        if self.selection_strategy == 'robust':
            # 选择访问次数最多的动作
            return max(root.children.items(),
                      key=lambda x: x[1].visits)[0]
        else:
            # 选择平均奖励最高的动作
            return max(root.children.items(),
                      key=lambda x: x[1].value / (x[1].visits + 1e-10))[0]

    def _evaluate_bridge_potential(self, board: Board, move: Action) -> float:
        """评估形成桥接的潜力"""
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        bridge_score = 0.0
        player = board.current_player
        
        # 检查每对相对方向
        for i in range(len(directions)//2):
            dir1, dir2 = directions[i], directions[i+len(directions)//2]
            x1, y1 = move.x + dir1[0], move.y + dir1[1]
            x2, y2 = move.x + dir2[0], move.y + dir2[1]
            
            # 检查两个方向是否都在棋盘内
            if (0 <= x1 < board.size and 0 <= y1 < board.size and
                0 <= x2 < board.size and 0 <= y2 < board.size):
                # 如果两个方向都是己方棋子，增加桥接分数
                if (board.board[x1, y1] == player and 
                    board.board[x2, y2] == player):
                    bridge_score += 1.0
                # 如果一个方向是己方棋子，另一个方向是空位
                elif ((board.board[x1, y1] == player and board.board[x2, y2] == 0) or
                      (board.board[x1, y1] == 0 and board.board[x2, y2] == player)):
                    bridge_score += 0.5
                
        return bridge_score / len(directions)

    def _evaluate_blocking_value(self, board: Board, move: Action) -> float:
        """评估一个动作的阻挡价值"""
        opponent = 3 - board.current_player
        blocking_score = 0.0
        
        # 临时模拟这步棋
        temp_board = board.copy()
        temp_board.board[move.x, move.y] = board.current_player
        
        # 检查这步棋是否阻断了对手的连接
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        for dx1, dy1 in directions:
            x1, y1 = move.x + dx1, move.y + dy1
            if not (0 <= x1 < board.size and 0 <= y1 < board.size):
                continue
                
            # 检查是否有对手的棋子
            if board.board[x1, y1] == opponent:
                # 检查这个对手棋子的连接情况
                connected_count = 0
                for dx2, dy2 in directions:
                    x2, y2 = x1 + dx2, y1 + dy2
                    if (0 <= x2 < board.size and 
                        0 <= y2 < board.size and 
                        board.board[x2, y2] == opponent):
                        connected_count += 1
                
                # 如果这步棋切断了对手的连接，增加阻挡分���
                if connected_count > 0:
                    blocking_score += connected_count / len(directions)
        
        return blocking_score / len(directions)

class Agent:
    """智能体"""
    def __init__(self, policy: Policy, estimator: Optional[ValueEstimator], 
                 player_id: int, name: str = ""):
        self.policy = policy
        self.estimator = estimator
        self.player_id = player_id
        self.name = name
        self.current_episode = Episode(player_id)
    
    def choose_action(self, board: Board) -> Action:
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
    
class GameExperiment:
    """戏实"""
    def __init__(self, board_size: int = 5):
        self.board = Board(board_size)
        self.agent1 = None
        self.agent2 = None
        self.logger = logging.getLogger(__name__)
    
    def set_agents(self, agent1: Agent, agent2: Agent):
        """设置对弈双方"""
        self.agent1 = agent1
        self.agent2 = agent2
    
    def play_game(self) -> Tuple[Agent, int]:
        """进行一局游戏，返获胜步数"""
        self.board.reset()
        current_agent = random.choice([self.agent1, self.agent2])
        moves_count = 0
        MAX_MOVES = 100
        
        while moves_count < MAX_MOVES:
            action = current_agent.choose_action(self.board)
            game_over, intermediate_reward = self.board.make_move(action)
            moves_count += 1
            
            if game_over:
                return current_agent, moves_count
            
            current_agent = self.agent2 if current_agent == self.agent1 else self.agent1
        
        return None, moves_count  # 平局

class ExperimentRunner:
    """实验运行器"""
    def __init__(self, total_rounds: int, statistics_rounds: int, num_cores: Optional[int] = None):
        self.total_rounds = total_rounds
        self.statistics_rounds = statistics_rounds
        self.num_cores = num_cores or (os.cpu_count() or 4)
        self.logger = logging.getLogger(__name__)
        # 添加超时设置
        self.timeout = 30  # 每个批次的超时时间（秒）
    
    def _run_game_batch(self, experiment: GameExperiment, num_games: int) -> Tuple[int, int]:
        """运行一批游戏并返回胜利统计"""
        wins1 = 0
        wins2 = 0
        
        # 添加超时检查
        start_time = time.time()
        
        for _ in range(num_games):
            # 检查是否超时
            if time.time() - start_time > self.timeout:
                self.logger.warning("Batch timeout reached")
                break
                
            try:
                winner, _ = experiment.play_game()
                if winner == experiment.agent1:
                    wins1 += 1
                elif winner == experiment.agent2:
                    wins2 += 1
            except Exception as e:
                self.logger.error(f"Game error: {e}")
                continue
                
        return wins1, wins2
        
    def run_experiment(self, experiment: GameExperiment) -> List[Tuple[float, float]]:
        """并行运行实验"""
        win_rates_history = []
        
        for round_start in range(0, self.total_rounds, self.statistics_rounds):
            total_wins = [0, 0]
            
            # 减小每个进程的负载
            games_per_process = min(100, self.statistics_rounds // self.num_cores)
            num_batches = self.statistics_rounds // games_per_process
            
            for batch in range(num_batches):
                # 使用较小的进程池
                with concurrent.futures.ProcessPoolExecutor(max_workers=min(4, self.num_cores)) as executor:
                    try:
                        # 设置超时
                        future_to_batch = {
                            executor.submit(
                                self._run_game_batch,
                                GameExperiment(experiment.board.size),
                                games_per_process
                            ): i for i in range(min(4, self.num_cores))
                        }
                        
                        # 收集结果，添加超时处理
                        for future in concurrent.futures.as_completed(future_to_batch, timeout=self.timeout):
                            try:
                                wins1, wins2 = future.result(timeout=5)  # 单个future的超时
                                total_wins[0] += wins1
                                total_wins[1] += wins2
                            except concurrent.futures.TimeoutError:
                                self.logger.warning("Future timeout")
                                continue
                            except Exception as e:
                                self.logger.error(f"Future error: {e}")
                                continue
                                
                    except Exception as e:
                        self.logger.error(f"Batch error: {e}")
                        continue
                    
                    # 清理进程池
                    executor.shutdown(wait=False)
            
            # 计算胜率
            total_games = sum(total_wins)
            if total_games > 0:
                win_rate1 = total_wins[0] / total_games
                win_rate2 = total_wins[1] / total_games
                win_rates_history.append((win_rate1, win_rate2))
                
                self.logger.info(f"Round {round_start + self.statistics_rounds}: "
                               f"{experiment.agent1.name} win rate = {win_rate1:.2f}, "
                               f"{experiment.agent2.name} win rate = {win_rate2:.2f}")
            
        return win_rates_history

def plot_comparison(results: Dict[str, List[Tuple[float, float]]], total_rounds: int, statistics_rounds: int):
    """绘制不同略的比较图"""
    plt.figure(figsize=(12, 6))
    
    # 计算x轴的合数
    rounds = list(range(statistics_rounds, 
                       total_rounds + 1, 
                       statistics_rounds))
    
    # 绘制每个策略的率曲线
    for label, win_rates in results.items():
        agent2_rates = [rate[1] for rate in win_rates]
        plt.plot(rounds, agent2_rates, label=label, marker='o')
    
    plt.xlabel('Rounds')
    plt.ylabel('Win Rate')
    plt.title('Strategy Comparison: MC vs Q-learning vs SARSA')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # 配置实验参数
    board_size = 5
    total_rounds = 10000
    statistics_rounds = 1000
    num_cores = 8  # 在这里配置使用的CPU核心数
    
    # MCTS配置
    mcts_configs = {
        'MCTS-Advanced': {
            'strategy': 'robust',
            'simulations': 3000,      # 总迭代次数保持3000
            'max_depth': 100,
            'c': 1.732,
            'use_rave': False,
            'base_rollouts_per_leaf': 20  # 基础rollout次数设为20
        }
    }
    
    # DynaQ配置
    config = LearningConfig(
        algorithm_type='DynaQ',
        initial_learning_rate=0.3,
        final_learning_rate=0.01,
        initial_epsilon=0.4,
        final_epsilon=0.05,
        gamma=0.99,
        planning_steps=300,
        batch_size=128,
        memory_size=100000
    )

    # 创建实验环境
    experiment = GameExperiment(board_size)
    runner = ExperimentRunner(total_rounds, statistics_rounds, num_cores)
    
    # 创算法配置
    base_config = {
        'epsilon': 0.1,
        'learning_rate': 0.1,
        'gamma': 0.95
    }
    
    # 设置对比实验
    results = {}
    
    # 添加性能监控
    start_time = time.time()
    
    # 添加MCTS
    print("\nRunning experiment with MCTS")
    experiment = GameExperiment(board_size)  # 确保创建了新的实验实例
    
    # 设置初始agents
    agent1 = Agent(RandomPolicy(), None, player_id=1, name="Random")
    agent2 = MCTSAgent(
        simulations_per_move=3000,
        max_depth=100,
        c=1.732,
        use_rave=False,
        selection_strategy='robust',
        base_rollouts_per_leaf=20,
        player_id=2,
        name="MCTS-Advanced"
    )
    experiment.set_agents(agent1, agent2)
    
    # 运行实验
    results["MCTS-Advanced"] = runner.run_experiment(experiment)
    
    # 添加DynaQ
    print("\nRunning experiment with DynaQ")
    config = LearningConfig(
        algorithm_type='DynaQ',
        initial_learning_rate=0.2,
        final_learning_rate=0.01,
        initial_epsilon=0.3,
        final_epsilon=0.05,
        gamma=0.99,
        planning_steps=200,        # 增加规划步数
        batch_size=64,
        memory_size=50000
    )
    algorithm = RLAlgorithm(config)
    algorithm.policy = UCBPolicy(algorithm.estimator, c=1.0)
    
    agent1 = Agent(RandomPolicy(), None, player_id=1, name="Random")
    agent2 = Agent(algorithm.policy, algorithm.estimator, 
                  player_id=2, name="DynaQ")
    
    experiment.set_agents(agent1, agent2)
    results["DynaQ"] = runner.run_experiment(experiment)
    
    # 绘制比较图
    plot_comparison(results, total_rounds, statistics_rounds)

class Evaluator:
    """模估器"""
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def evaluate_episode(self, episode: Episode, board: Board) -> Dict[str, float]:
        """评估一局游戏的表现"""
        metrics = {
            'episode_length': len(episode.states),
            'final_reward': episode.reward,
            'avg_q_value': self._calculate_avg_q_value(episode),
            'connectivity_score': self._calculate_connectivity_score(board)
        }
        return metrics

class TrainingVisualizer:
    """训过程化"""
    def __init__(self):
        self.metrics_history = defaultdict(list)
    
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
    
    def plot_metrics(self):
        """绘制训练指标变化趋势"""
        plt.figure(figsize=(15, 5))
        for key, values in self.metrics_history.items():
            plt.plot(values, label=key)
        plt.legend()
        plt.show()

class Model:
    """环境模型，用于Dyna-Q的拟"""
    def __init__(self):
        self.state_actions = {}  # 存(state, action) -> (next_state, reward)的映射
    
    def update(self, state: State, action: Action, next_state: State, reward: float):
        """更新模型"""
        self.state_actions[(state, action)] = (next_state, reward)
    
    def sample(self) -> Optional[Tuple[State, Action, State, float]]:
        """从模型中随机样一个经验"""
        if not self.state_actions:
            return None
        state, action = random.choice(list(self.state_actions.keys()))
        next_state, reward = self.state_actions[(state, action)]
        return state, action, next_state, reward

class DynaQEstimator(ValueEstimator):
    """Dyna-Q值函数估计器"""
    def __init__(self, config: LearningConfig):
        super().__init__(config)
        self.model = Model()
        self.planning_steps = config.planning_steps
    
    def update(self, episode: Episode, board: Board):
        """更新值函数和模型"""
        for i in range(len(episode.states) - 1):
            state = episode.states[i]
            action = episode.actions[i]
            next_state = episode.states[i + 1]
            reward = episode.reward
            
            # 更新模型
            self.model.update(state, action, next_state, reward)
            
            # 直接经验更新
            self._update_q_value(state, action, reward, next_state, board)
            
            # 规划更新
            self._planning(board)
    
    def _planning(self, board: Board):
        """使用模型进行规划"""
        for _ in range(self.planning_steps):
            experience = self.model.sample()
            if experience is None:
                continue
                
            state, action, next_state, reward = experience
            self._update_q_value(state, action, reward, next_state, board)

class MCTSAgent(Agent):
    """使用MCTS的智能体"""
    def __init__(self, 
                 simulations_per_move: int = 50, 
                 max_depth: int = 50,
                 c: float = 1.414, 
                 selection_strategy: str = 'robust',
                 use_rave: bool = False,
                 base_rollouts_per_leaf: int = 20,
                 player_id: int = 1, 
                 name: str = "MCTS"):
        policy = MCTSPolicy(
            simulations_per_move=simulations_per_move,
            max_depth=max_depth,
            c=c,
            selection_strategy=selection_strategy,
            use_rave=use_rave,
            base_rollouts_per_leaf=base_rollouts_per_leaf,
            player_id=player_id
        )
        super().__init__(policy, None, player_id, name)

class RLAlgorithm:
    """强化学习算法"""
    def __init__(self, config: LearningConfig):
        self.config = config
        self.estimator = self._create_estimator()
        self.policy = self._create_policy()
    
    def _create_estimator(self) -> ValueEstimator:
        """创建值函数估计器"""
        if self.config.algorithm_type == 'MC':
            return MonteCarloEstimator(self.config)
        elif self.config.algorithm_type == 'QLearning':
            return QLearningEstimator(self.config)
        elif self.config.algorithm_type == 'SARSA':
            return SarsaEstimator(self.config)
        elif self.config.algorithm_type == 'DynaQ':
            return DynaQEstimator(self.config)
        else:
            raise ValueError(f"Unknown algorithm type: {self.config.algorithm_type}")
    
    def _create_policy(self) -> Policy:
        """创建策略"""
        return GreedyPolicy(self.estimator, self.config.epsilon)

if __name__ == "__main__":
    # 设置进启动方法（在 main 函数开始处添加）
    if sys.platform == 'darwin':  # macOS
        multiprocessing.set_start_method('spawn')
    
    main() 