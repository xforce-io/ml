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
                
                # 根据游戏阶段使用不同的策略
                action = self._get_simulation_action(board, valid_moves, player, depth)
                
                game_over, reward = board.make_move(action, player)
                depth += 1
                
                if game_over:
                    # 计算相对于根节点玩家的奖励
                    final_reward = 1.0 if player == self.player_id else -1.0
                    rewards.append(final_reward)
                    break
                
                player = 3 - player  # 切换玩家
            
            if depth >= self.config.max_depth:
                position_value = self._evaluate_position(board, self.player_id)
                rewards.append(position_value)
        
        return sum(rewards) / len(rewards)
    
    def _get_simulation_action(self, board: Board, valid_moves: List[Action], player: int, depth: int) -> Action:
        """获取模拟阶段的动作
        
        Args:
            board: 当前棋盘
            valid_moves: 有效动作列表
            player: 当前玩家
            depth: 当前深度
            
        Returns:
            Action: 选择的动作
        """
        # 计算每个动作的分数
        move_scores = []
        for move in valid_moves:
            score = 0.0
            
            # 1. 连接性评分 (0-1)
            connectivity = self._evaluate_connectivity(board, move, player)
            score += 0.25 * connectivity
            
            # 2. 到目标的距离评分 (0-1)
            distance_score = self._evaluate_distance_to_goal(board, move, player)
            score += 0.2 * distance_score
            
            # 3. 桥接潜力评分 (0-1)
            bridge_score = self._evaluate_bridge_potential(board, move, player)
            score += 0.2 * bridge_score
            
            # 4. 阻挡对手评分 (0-1)
            blocking_score = self._evaluate_blocking_value(board, move, player)
            score += 0.15 * blocking_score
            
            # 5. 虚桥评分 (0-1) - 新增
            virtual_bridge_score = self._evaluate_virtual_bridge(board, move, player)
            score += 0.1 * virtual_bridge_score
            
            # 6. 中心控制评分 (0-1) - 新增
            center_score = self._evaluate_center_control(board, move, player)
            score += 0.1 * center_score
            
            # 添加随机扰动以增加多样性，但减小扰动范围
            score += random.uniform(0, 0.05)
            
            move_scores.append((score, move))
        
        # 根据游戏阶段和局面特征选择不同的策略
        empty_spaces = len(valid_moves)
        total_spaces = board.size * board.size
        game_progress = 1.0 - (empty_spaces / total_spaces)
        
        if game_progress < 0.3:  # 开局阶段
            # 选择分数最高的几个动作中的一个
            move_scores.sort(key=lambda x: x[0], reverse=True)
            top_k = max(1, min(3, len(move_scores) // 4))
            return move_scores[random.randint(0, top_k-1)][1]
        elif game_progress < 0.7:  # 中盘
            # 使用 softmax 选择，温度参数随游戏进程调整
            temperature = 1.0 - game_progress
            scores = np.array([s[0] for s in move_scores])
            scores = np.exp(scores / temperature - np.max(scores) / temperature)
            probs = scores / scores.sum()
            idx = np.random.choice(len(move_scores), p=probs)
            return move_scores[idx][1]
        else:  # 残局
            # 更倾向于选择最优动作
            move_scores.sort(key=lambda x: x[0], reverse=True)
            if random.random() < 0.8:  # 80%概率选择最佳动作
                return move_scores[0][1]
            else:  # 20%概率从前三个动作中随机选择
                top_k = min(3, len(move_scores))
                return move_scores[random.randint(0, top_k-1)][1]
    
    def _evaluate_distance_to_goal(self, board: Board, move: Action, player: int) -> float:
        """评估一个动作到目标边的距离
        
        Returns:
            float: 归一化的距离分数 (0-1)，越近分数越高
        """
        size = board.size
        if player == 1:  # 横向连接
            distance = size - 1 - move.y
        else:  # 纵向连接
            distance = size - 1 - move.x
        
        # 归一化到 0-1 范围，并反转使得距离越近分数越高
        return 1.0 - (distance / (size - 1))
    
    def _evaluate_blocking_value(self, board: Board, move: Action, player: int) -> float:
        """评估一个动作的阻挡价值
        
        Returns:
            float: 阻挡价值分数 (0-1)
        """
        opponent = 3 - player
        opponent_pieces = 0
        total_neighbors = 0
        
        # 检查周围六个方向
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        for dx, dy in directions:
            x, y = move.x + dx, move.y + dy
            if 0 <= x < board.size and 0 <= y < board.size:
                total_neighbors += 1
                if board.board[x, y] == opponent:
                    opponent_pieces += 1
        
        return opponent_pieces / max(1, total_neighbors)
    
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

    def _evaluate_virtual_bridge(self, board: Board, move: Action, player: int) -> float:
        """评估形成虚桥的潜力
        
        虚桥是一种特殊的连接模式，可以保证连接的形成
        """
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        virtual_bridges = 0
        total_patterns = 0
        
        # 检查所有可能的虚桥模式
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                dir1, dir2 = directions[i], directions[j]
                x1, y1 = move.x + dir1[0], move.y + dir1[1]
                x2, y2 = move.x + dir2[0], move.y + dir2[1]
                
                if (0 <= x1 < board.size and 0 <= y1 < board.size and
                    0 <= x2 < board.size and 0 <= y2 < board.size):
                    total_patterns += 1
                    
                    # 检查是否形成虚桥模式
                    if (board.board[x1, y1] == 0 and board.board[x2, y2] == 0 and
                        self._check_virtual_bridge_pattern(board, move, (x1,y1), (x2,y2), player)):
                        virtual_bridges += 1
        
        return virtual_bridges / max(1, total_patterns)

    def _check_virtual_bridge_pattern(self, board: Board, move: Action, 
                                    pos1: Tuple[int,int], pos2: Tuple[int,int], 
                                    player: int) -> bool:
        """检查是否形成有效的虚桥模式"""
        # 检查两个位置之间是否能形成有效的连接
        x1, y1 = pos1
        x2, y2 = pos2
        
        # 检查是否有对手的棋子阻挡
        opponent = 3 - player
        if any(board.board[x, y] == opponent 
               for x, y in [(x1,y1), (x2,y2)]):
            return False
        
        # 检查是否与己方棋子相邻
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        has_friendly_neighbor = False
        
        for dx, dy in directions:
            x, y = move.x + dx, move.y + dy
            if (0 <= x < board.size and 0 <= y < board.size and 
                board.board[x, y] == player):
                has_friendly_neighbor = True
                break
        
        return has_friendly_neighbor

    def _evaluate_center_control(self, board: Board, move: Action, player: int) -> float:
        """评估中心控制价值
        
        中心位置通常具有更高的战略价值
        """
        size = board.size
        center_x = size // 2
        center_y = size // 2
        
        # 计算到中心的距离
        dx = abs(move.x - center_x)
        dy = abs(move.y - center_y)
        manhattan_dist = dx + dy
        
        # 归一化距离分数，距离越近分数越高
        max_dist = size - 1
        distance_score = 1.0 - (manhattan_dist / max_dist)
        
        # 考虑周围己方棋子的影响
        friendly_pieces = 0
        directions = [(0,1), (1,0), (-1,0), (0,-1), (1,-1), (-1,1)]
        for dx, dy in directions:
            x, y = move.x + dx, move.y + dy
            if (0 <= x < size and 0 <= y < size and 
                board.board[x, y] == player):
                friendly_pieces += 1
        
        friendly_score = friendly_pieces / len(directions)
        
        # 综合评分
        return 0.7 * distance_score + 0.3 * friendly_score

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
        (1, 2), (1, 3), (1, 4),
    ]
    
    player2_moves = [
        (3, 0), (3, 2), (4, 1),
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