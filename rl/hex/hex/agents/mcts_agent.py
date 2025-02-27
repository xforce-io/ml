from __future__ import annotations
from abc import abstractmethod
import logging
import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Action:
    """动作"""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

from hex.agents.agent import Agent
from hex.hex import Board, State
from hex.config import DEBUG_MCTS, MCTSConfig
from hex.log import DEBUG, INFO, ERROR
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
        """节点评估，使用UCT公式并结合先验概率和价值预测"""
        if self.visits == 0:
            return self.value_prediction if self.value_prediction is not None else float('inf')
            
        # 计算基础UCT分数
        mc_score = self.value / self.visits
        
        # 如果有价值预测，将其与MC分数结合
        if self.value_prediction is not None:
            mc_score = 0.8 * mc_score + 0.2 * self.value_prediction
        
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

    def paint(self) -> str:
        """以树形结构打印节点信息
        
        Returns:
            str: 格式化的树形字符串
        """
        def _node_info(node: MCTSNode) -> str:
            """生成节点信息字符串"""
            if node.action:
                action_str = f"({node.action.x},{node.action.y})"
            else:
                action_str = "root"
            
            value = node.value / node.visits if node.visits > 0 else 0
            info = f"{action_str} p:{node.state.current_player} v:{value:.3f} n:{node.visits}"
            
            if node.use_rave and node.rave_visits > 0:
                rave_value = node.rave_value / node.rave_visits
                info += f" rv:{rave_value:.3f} rn:{node.rave_visits}"
            
            return info
        
        def _paint_tree(node: MCTSNode, prefix: str = "", is_last: bool = True) -> str:
            """递归生成树形结构字符串"""
            result = prefix
            result += "└── " if is_last else "├── "
            result += _node_info(node) + "\n"
            
            children = list(node.children.values())
            for i, child in enumerate(children):
                extension = "    " if is_last else "│   "
                result += _paint_tree(child, prefix + extension, i == len(children)-1)
            
            return result
        
        return _paint_tree(self)

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
        self.use_network = False
        self.prior_probs = None
        self.value_estimate = None

    def search(self, board: Board, num_simulations, use_network: bool) -> np.ndarray:
        """执行MCTS搜索并返回动作概率分布
        
        Args:
            board: 当前棋盘状态
            num_simulations: 可选的模拟次数
            
        Returns:
            np.ndarray: 所有可能位置的动作概率
        """
        DEBUG(logger, f"search, board: {board.get_state(self.player_id).board} num_simulations: {num_simulations}")

        self.use_network = use_network
            
        root = MCTSNode(board.get_state(self.player_id), use_rave=self.config.use_rave)
        root.untried_actions = board.get_valid_moves()
        
        for i in range(num_simulations):
            if DEBUG_MCTS:
                print(f"\nsimulation {i}/{num_simulations}")

            board_copy = board.copy()
            self._run_simulation(root, board_copy)

        self.root = root
        return self.get_action_probs(board)

    def select_action(self, board: Board, action_probs: np.ndarray) -> Action:
        """选择概率最大的动作
        
        Args:
            board: 当前棋盘
            action_probs: 所有位置的动作概率分布
            
        Returns:
            选择的动作
            
        Note:
            如果概率分布无效，会回退到随机选择
        """
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
            
        # 提取合法动作的概率
        move_probs = [(move, action_probs[move.x * board.size + move.y]) 
                     for move in valid_moves]
        
        best_move, best_prob = max(move_probs, key=lambda x: x[1])
        return best_move

    def search_and_select_action(self, board: Board, use_network: bool) -> Action:
        """执行搜索并选择动作
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            Action: 选择的动作
        """
        probs = self.search(board, self.config.simulations, use_network)
        action = self.select_action(board, probs)
        self.reset()  # 重置搜索树
        return action

    def get_action_probs(self, board: Board) -> np.ndarray:
        """获取基于访问次数的动作概率分布"""
        if not hasattr(self, 'root'):
            raise ValueError("Must call search() before get_action_probs()")
            
        visits = np.zeros(board.size * board.size)
        
        # 收集所有子节点的访问次数
        for action, child in self.root.children.items():
            idx = action.x * board.size + action.y
            visits[idx] = child.visits
            
        # 归一化得到概率分布
        total_visits = visits.sum()
        if total_visits > 0:
            DEBUG(logger, f"visits: {visits}, total_visits: {total_visits}")
            probs = visits / total_visits
        else:
            DEBUG(logger, f"no visits")
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
        """运行单次模拟
        
        在树搜索过程中，玩家会交替行动。每一层的节点代表当前行动玩家的选择。
        """
        node = root
        current_player = self.player_id
        
        if DEBUG_MCTS:
            print(f"before selection, board \n {board.paint(node.action)}")
        
        # Selection - 在每一层交替玩家
        while not node.is_terminal() and node.is_fully_expanded():
            node = self._select_child(node)
            board.make_move(node.action, current_player)

            if DEBUG_MCTS:
                print(f"selection, player {current_player}, action: {node.action}")

            current_player = 3 - current_player  # 切换玩家
        
        # Expansion - 扩展时使用当前玩家
        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node, board)

            if DEBUG_MCTS:
                print(f"expand, player {current_player}, action: {node.action}")

            current_player = 3 - current_player  # 切换玩家

            if DEBUG_MCTS:
                print(f"after expand, board \n {board.paint(node.action)}")
        
        # Simulation - 从当前玩家开始模拟
        reward = self._simulate(board, current_player)

        if DEBUG_MCTS:
            print(f"after simulate, reward: {reward}")
        
        # Backpropagation - 注意reward的视角是相对于root玩家的
        self._backpropagate(node, reward)

        if DEBUG_MCTS:
            print(f"after backpropagation, root \n {root.paint()}")
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """选子节点"""
        return max(node.children.values(), 
                  key=lambda n: n.get_value(self.config.c, self.config.rave_constant))
    
    def _expand(self, node: MCTSNode, board: Board) -> MCTSNode:
        """扩展节点，结合策略网络和随机选择
        
        Args:
            node: 要扩展的节点
            board: 当前棋盘状态
            
        Returns:
            MCTSNode: 新创建的子节点
        """
        action = None
        value = None
        current_player = node.state.current_player  # 获取当前玩家
        
        if self.use_network and self.state_predictor is not None:
            # 获取预测值
            probs, value = self.state_predictor.predict(node.state)
            
            # 获取所有未尝试动作的策略预测值
            valid_probs = []
            for act in node.untried_actions:
                idx = act.x * self.board_size + act.y
                valid_probs.append((probs[idx], idx, act))  # (概率, 索引, 动作)
            
            if valid_probs:
                valid_probs.sort(key=lambda x: (-x[0], x[1]))
                top_k = max(1, min(3, len(valid_probs) // 4))
                action = valid_probs[random.randint(0, top_k-1)][2]
                node.untried_actions.remove(action)
        
        # 如果没有使用策略网络或随机选择不使用策略网络
        if action is None and node.untried_actions:  # 确保有未尝试的动作
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
        
        # 如果没有可用动作，返回None或抛出异常
        if action is None:
            raise ValueError("No valid actions available for expansion")
        
        DEBUG(logger, f"expand, action: {action}")
        board.make_move(action, current_player)  # 使用当前玩家执行动作
        
        # 创建新的子节点状态时，需要从对手的视角创建
        next_player = 3 - current_player  # 切换到下一个玩家
        child_state = board.get_state(next_player)  # 从下一个玩家的视角创建状态
        
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
    
    def _simulate(self, board: Board, current_player: int) -> float:
        """使用神经网络的价值估计或随机模拟
        
        Args:
            board: 当前棋盘状态
            current_player: 当前模拟的玩家
            
        Returns:
            float: 从根节点玩家(self.player_id)视角看的奖励值
        """
        if self.use_network and self.state_predictor is not None:
            _, value = self.state_predictor.predict(board.get_state(self.player_id))
            return value
            
        rewards = []
        original_board = board.copy()
        rollouts = self._get_dynamic_rollouts(original_board)
        
        for _ in range(rollouts):
            board = original_board.copy()
            player = current_player  # 从当前玩家开始模拟
            depth = 0
            
            while depth < self.config.max_depth:
                valid_moves = board.get_valid_moves()
                if not valid_moves:
                    rewards.append(0.0)
                    break
                
                if depth < self.config.max_depth // 4:
                    action = self._get_early_game_action(board, valid_moves)
                elif depth < self.config.max_depth * 3 // 4:
                    if random.random() < 0.8:
                        action = self._get_heuristic_action(board, valid_moves, player)
                    else:
                        action = random.choice(valid_moves)
                else:
                    action = self._get_endgame_action(board, valid_moves, player)
                
                game_over, reward = board.make_move(action, player)
                depth += 1
                
                if game_over:
                    # 计算相对于根节点玩家的奖励
                    # 如果当前玩家赢了，那么reward就是1.0，这时我们需要判断这个玩家是否是根节点玩家
                    final_reward = 1.0 if player == self.player_id else -1.0
                    rewards.append(final_reward)
                    break
                
                player = 3 - player  # 切换玩家
            
            if depth >= self.config.max_depth:
                position_value = self._evaluate_position(board, self.player_id)
                # 如果当前玩家不是根节点玩家，需要翻转评估值
                if current_player != self.player_id:
                    position_value = -position_value
                rewards.append(position_value)
        
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
        return self._get_heuristic_action(board, valid_moves, self.player_id)
    
    def _get_endgame_action(self, board: Board, valid_moves: List[Action], current_player: int) -> Action:
        """残局策略"""
        best_score = float('-inf')
        best_moves = []
        
        for move in valid_moves:
            temp_board = board.copy()
            temp_board.make_move(move, current_player)
            
            # 评估这步棋后的局面
            score = self._evaluate_position(temp_board, current_player)
            
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        
        return random.choice(best_moves)
    
    def _get_heuristic_action(self, board: Board, valid_moves: List[Action], current_player: int) -> Action:
        """改进的启发式动作选择"""
        move_scores = []
        center = board.size // 2
        
        for move in valid_moves:
            # 计算多个特征
            distance_to_center = abs(move.x - center) + abs(move.y - center)
            connectivity_score = self._evaluate_connectivity(board, move, current_player)
            bridge_score = self._evaluate_bridge_potential(board, move, current_player)
            blocking_score = self._evaluate_blocking_value(board, move, current_player)
            
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
    
    def _evaluate_connectivity(self, board: Board, move: Action, current_player: int) -> float:
        """评估一个动作的连接价值"""
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        connected_pieces = 0
        
        for dx, dy in directions:
            x, y = move.x + dx, move.y + dy
            if (0 <= x < board.size and 0 <= y < board.size and 
                board.board[x, y] == current_player):
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
        """反向传播
        
        Args:
            node: 开始反向传播的节点
            reward: 从根节点玩家视角看到的奖励值
        """
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
        for path_node in path:
            # 直接更新节点统计，不需要翻转reward
            path_node.update(reward)
            DEBUG(logger, f"backpropagate, action: {path_node.action}, reward: {reward}")
            
            # RAVE更新
            if path_node.use_rave and path_node.parent:
                children = path_node.parent.get_children()
                for action, child in children.items():
                    if action in actions_seen:
                        child.update_rave(reward)

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

    def _evaluate_bridge_potential(self, board: Board, move: Action, current_player: int) -> float:
        """评估形成桥接潜力"""
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        bridge_score = 0.0
        
        for i in range(len(directions)//2):
            dir1, dir2 = directions[i], directions[i+len(directions)//2]
            x1, y1 = move.x + dir1[0], move.y + dir1[1]
            x2, y2 = move.x + dir2[0], move.y + dir2[1]
            
            if (0 <= x1 < board.size and 0 <= y1 < board.size and
                0 <= x2 < board.size and 0 <= y2 < board.size):
                if (board.board[x1, y1] == current_player and 
                    board.board[x2, y2] == current_player):
                    bridge_score += 1.0
                elif ((board.board[x1, y1] == current_player and board.board[x2, y2] == 0) or
                      (board.board[x1, y1] == 0 and board.board[x2, y2] == current_player)):
                    bridge_score += 0.5
                
        return bridge_score / len(directions)

    def _evaluate_blocking_value(self, board: Board, move: Action, current_player: int) -> float:
        """评估一个动作的阻挡价值"""
        opponent = 3 - current_player
        blocking_score = 0.0
        
        temp_board = board.copy()
        temp_board.board[move.x, move.y] = current_player
        
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        for dx1, dy1 in directions:
            x1, y1 = move.x + dx1, move.y + dy1
            if not (0 <= x1 < board.size and 0 <= y1 < board.size):
                continue
                
            if board.board[x1, y1] == opponent:
                connected_count = 0
                for dx2, dy2 in directions:
                    x2, y2 = x1 + dx2, y1 + dy2
                    if (0 <= x2 < board.size and 
                        0 <= y2 < board.size and 
                        board.board[x2, y2] == opponent):
                        connected_count += 1
                
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

if __name__ == "__main__":
    import numpy as np
    from hex.hex import Board, Action
    from hex.config import MCTSConfig
    import logging
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    random.seed(1)
    
    board = Board(size=5)
    
    player1_moves = [
        (0, 2), (1, 1), (1, 2), (1, 4), (2, 2), (2, 4), (3, 0), (3,4), (4, 0),
    ]
    
    player2_moves = [
        (0, 0), (0, 1), (2, 0), (2, 3), (3, 1), (3, 3), (4, 1), (4, 2), (4, 3)
    ]
    
    # 放置棋子
    for move in player1_moves:
        board.make_move(Action(*move), 1)
    for move in player2_moves:
        board.make_move(Action(*move), 2)
        
    # 创建MCTS策略
    config = MCTSConfig()
    policy = MCTSPolicy(config=config, board_size=5, player_id=1)
    
    # 执行搜索
    action_probs = policy.search(board, num_simulations=config.simulations, use_network=False)
    action = policy.select_action(board, action_probs)
    
    # 打印棋盘状态
    print("\n当前棋盘状态:")
    print(board.paint(action, action_probs))