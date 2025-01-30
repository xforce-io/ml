from __future__ import annotations
from abc import abstractmethod
import logging
import math
import random
import time
from typing import Dict, List, Optional, Tuple
from hex.agents.agent import Agent
from hex.hex import Action, Board, State
from hex.config import MCTSConfig
from hex.log import INFO, ERROR
from hex.rl_basic import Policy
import numpy as np

logger = logging.getLogger(__name__)

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
        
        self.prior_prob = 0.0  # 添加先验概率属性
        self.value_prediction = None  # 添加价值预测属性
    
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
        """节点评估，结合先验概率和价值预测"""
        if self.visits == 0:
            # 如果有价值预测，在未访问时使用它
            return self.value_prediction if self.value_prediction is not None else float('inf')
            
        mc_score = self.value / self.visits
        # 如果有价值预测，将其与MC分数结合
        if self.value_prediction is not None:
            mc_score = 0.8 * mc_score + 0.2 * self.value_prediction
            
        # 加入先验概率的影响
        exploration = c * self.prior_prob * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        if not self.use_rave:
            return mc_score + exploration
            
        # RAVE评估
        beta = (rave_constant / (3 * self.visits + rave_constant)) ** 2
        rave_score = self.rave_value / (self.rave_visits + 1e-5)
        return (1 - beta) * (mc_score + exploration) + beta * rave_score
    
    def get_children(self) -> Dict[Action, MCTSNode]:
        """安全地获取children"""
        return dict(self.children)  # 返回副本

class StatePredictor:
    @abstractmethod
    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        pass

class MCTSPolicy(Policy):
    """基于MCTS的策略"""
    def __init__(self, 
                 config: MCTSConfig,
                 board_size: int,
                 num_threads: int = 1,
                 player_id: int = 1,
                 state_predictor: StatePredictor = None):
        self.config = config    
        self.board_size = board_size
        self.num_threads = num_threads
        self.player_id = player_id
        self.state_predictor = state_predictor
        self.prior_probs = None
        self.value_estimate = None

    def search(self, board: Board, num_simulations: int = None) -> np.ndarray:
        """执行MCTS搜索并返回动作概率分布
        
        Args:
            board: 当前棋盘状态
            num_simulations: 可选的模拟次数
            
        Returns:
            np.ndarray: 所有可能位置的动作概率
        """
        if num_simulations is None:
            num_simulations = self.config.simulations
            
        root = MCTSNode(board.get_state(), use_rave=self.config.use_rave)
        root.untried_actions = board.get_valid_moves()
        
        for _ in range(num_simulations):
            board_copy = board.copy()
            self._run_simulation(root, board_copy)
            
        self.root = root
        action_probs = self.get_action_probs(board)
        return action_probs

    def select_action(self, board: Board, action_probs: np.ndarray, temperature: float = 1.0) -> Action:
        """根据动作概率选择动作"""
        valid_moves = board.get_valid_moves()
        
        # 提取合法动作的概率
        valid_probs = []
        for move in valid_moves:
            idx = move.x * board.size + move.y
            valid_probs.append(action_probs[idx])
        
        valid_probs = np.array(valid_probs)
        
        # 应用温度
        if temperature != 1.0:
            valid_probs = np.power(valid_probs, 1.0 / temperature)
        
        # 确保概率和为1
        valid_probs = valid_probs / np.sum(valid_probs)
        
        # 如果概率和太小，使用均匀分布
        if np.sum(valid_probs) < 1e-6:
            valid_probs = np.ones(len(valid_moves)) / len(valid_moves)
        
        try:
            action_idx = np.random.choice(len(valid_moves), p=valid_probs)
            return valid_moves[action_idx]
        except ValueError as e:
            # 如果仍然出现问题，记录详细信息并使用均匀分布
            ERROR(logger, f"Error in select_action: {e}, valid_probs={valid_probs}, sum={np.sum(valid_probs)}")
            action_idx = np.random.choice(len(valid_moves))
            return valid_moves[action_idx]

    def search_and_select_action(self, board: Board, temperature: float = 1.0) -> Action:
        """执行搜索并选择动作
        
        Args:
            board: 当前棋盘状态
            temperature: 温度参数
            
        Returns:
            Action: 选择的动作
        """
        probs = self.search(board)
        action = self.select_action(board, probs, temperature)
        self.reset()  # 重置搜索树
        return action

    def get_action_probs(self, board: Board, temperature: float = 1.0) -> np.ndarray:
        """获取基于访问次数的动作概率分布"""
        if not hasattr(self, 'root'):
            raise ValueError("Must call search() before get_action_probs()")
            
        visits = np.zeros(board.size * board.size)
        
        # 收集所有子节点的访问次数
        for action, child in self.root.children.items():
            idx = action.x * board.size + action.y
            visits[idx] = child.visits
            
        if temperature == 0:
            # 选择访问次数最多的动作
            best_idx = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[best_idx] = 1.0
            return probs
            
        # 应用温度
        if temperature != 1.0:
            visits = np.power(visits, 1.0/temperature)
            
        # 归一化得到概率分布
        total_visits = visits.sum()
        if total_visits > 0:
            probs = visits / total_visits
        else:
            # 如果没有访问记录，使用均匀分布
            valid_moves = board.get_valid_moves()
            probs = np.zeros_like(visits)
            for move in valid_moves:
                idx = move.x * board.size + move.y
                probs[idx] = 1.0 / len(valid_moves)
                
        return probs

    def reset(self):
        """重置搜索树"""
        if hasattr(self, 'root'):
            delattr(self, 'root')
        self.prior_probs = None
        self.value_estimate = None

    # 以下是私有方法...
    def _run_simulation(self, root: MCTSNode, board: Board) -> None:
        """运行单次模拟"""
        node = root
        
        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = self._select_child(node)
            board.make_move(node.action)
        
        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node, board)
        
        # Simulation
        reward = self._simulate(board)
        
        # Backpropagation
        self._backpropagate(node, reward)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """选子节点"""
        return max(node.children.values(), 
                  key=lambda n: n.get_value(self.config.c, self.config.rave_constant))
    
    def _expand(self, node: MCTSNode, board: Board) -> MCTSNode:
        """扩展节点，结合策略网络和随机选择"""
        action = None
        value = None
        
        if self.state_predictor is not None:
            # 获取预测值
            probs, value = self.state_predictor.predict(node.state)
            
            if random.random() < 0.3:  # 30%概率使用策略网络
                # 获取所有未尝试动作的策略预测值
                valid_probs = []
                for act in node.untried_actions:
                    idx = act.x * self.board_size + act.y
                    valid_probs.append((probs[idx], idx, act))  # 添加idx作为稳定的第二排序键
                
                if valid_probs:  # 确保有有效动作
                    # 按概率排序
                    valid_probs.sort(key=lambda x: (-x[0], x[1]))  # 使用负概率实现降序，idx作为次要排序键
                    # 确保top_k至少为1
                    top_k = max(1, min(3, len(valid_probs) // 4))  # 取前25%的动作，但至少1个，最多3个
                    action = valid_probs[random.randint(0, top_k-1)][2]  # 获取Action对象
                    node.untried_actions.remove(action)
        
        # 如果没有使用策略网络或随机选择不使用策略网络
        if action is None and node.untried_actions:  # 确保有未尝试的动作
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
        
        # 如果没有可用动作，返回None或抛出异常
        if action is None:
            raise ValueError("No valid actions available for expansion")
        
        board.make_move(action)
        child_state = board.get_state()
        child = MCTSNode(child_state, parent=node, action=action, 
                         use_rave=self.config.use_rave)
        
        # 保存价值预测
        if value is not None:
            child.value_prediction = value
        
        child.untried_actions = board.get_valid_moves()
        node.children[action] = child
        return child
    
    def _get_dynamic_rollouts(self, board: Board) -> int:
        """根据剩余空格动态调整rollout次数"""
        empty_spaces = len(board.get_valid_moves())
        total_spaces = board.size * board.size
        
        # 根据剩余空格比例调整rollout次数
        # 游戏后期（空格少）时增加rollout次数
        ratio = 1.0 - (empty_spaces / total_spaces)  # 比例从0到1
        additional_rollouts = int(10 * ratio)  # 最多额外增加10次
        
        return self.config.base_rollouts_per_leaf + additional_rollouts
    
    def _simulate(self, board: Board) -> float:
        """使用神经网络的价值估计替代随机模拟"""
        # 如果有神经网络的价值估计，直接使用
        if self.state_predictor is not None:
            _, value = self.state_predictor.predict(board.get_state())
            return value
            
        # 否则回退到传统的随机模拟
        rewards = []
        original_board = board.copy()
        rollouts = self._get_dynamic_rollouts(original_board)
        
        # 对同一个叶子节点进行多次rollout
        for _ in range(rollouts):
            board = original_board.copy()
            original_player = board.current_player
            depth = 0
            
            while depth < self.config.max_depth:
                valid_moves = board.get_valid_moves()
                if not valid_moves:
                    rewards.append(0.0)
                    break
                
                # 动态调整策略
                if depth < self.config.max_depth // 4:  # 开局阶段
                    action = self._get_early_game_action(board, valid_moves)
                elif depth < self.config.max_depth * 3 // 4:  # 中局阶段
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
            
            if depth >= self.config.max_depth:
                rewards.append(self._evaluate_position(board, original_player))
        
        # 返回所有rollout的平均奖励
        return sum(rewards) / len(rewards)
    
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
        """计算中心区域控制度"""
        size = board.size
        center = size // 2
        center_score = 0.0
        total_weights = 0.0
        
        for x in range(size):
            for y in range(size):
                if board.board[x, y] == player:
                    # 距离中心越权重越大
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

    def _evaluate_bridge_potential(self, board: Board, move: Action) -> float:
        """评估形成桥接潜力"""
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
                # 检查这个手棋子的连接情况
                connected_count = 0
                for dx2, dy2 in directions:
                    x2, y2 = x1 + dx2, y1 + dy2
                    if (0 <= x2 < board.size and 
                        0 <= y2 < board.size and 
                        board.board[x2, y2] == opponent):
                        connected_count += 1
                
                # 如果这步棋切断了对手的连接，增加阻挡分
                if connected_count > 0:
                    blocking_score += connected_count / len(directions)
        
        return blocking_score / len(directions)

class MCTSAgent(Agent):
    """MCTS智能体"""
    def __init__(self, 
                 simulations_per_move: int = 3000,
                 max_depth: int = 50,
                 c: float = 1.414,
                 use_rave: bool = False,
                 selection_strategy: str = 'robust',
                 base_rollouts_per_leaf: int = 20,
                 player_id: int = 1,
                 name: str = "MCTS"):
        # 创建MCTS策略
        policy = MCTSPolicy(
            simulations_per_move=simulations_per_move,
            max_depth=max_depth,
            c=c,
            use_rave=use_rave,
            selection_strategy=selection_strategy,
            base_rollouts_per_leaf=base_rollouts_per_leaf,
            player_id=player_id
        )
        # 调用父类构造函数
        super().__init__(
            policy=policy,
            estimator=None,  # MCTS不需要估计器
            player_id=player_id,
            name=name
        )

def create_mcts_agent(config: MCTSConfig, player_id: int) -> MCTSAgent:
    """创建MCTS智能体"""
    return MCTSAgent(
        simulations_per_move=config.simulations,
        max_depth=config.max_depth,
        c=config.c,
        use_rave=config.use_rave,
        selection_strategy=config.strategy,
        base_rollouts_per_leaf=config.base_rollouts_per_leaf,
        player_id=player_id,
        name=config.name
    )