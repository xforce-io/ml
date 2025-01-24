from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

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
    
    def __hash__(self):
        return hash((self.board.tobytes(), self.current_player))
    
    def __eq__(self, other):
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)
    
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
        # 使用numpy的向量化操作找到所有空位置
        empty_positions = np.where(self.board == 0)
        moves = []
        
        for x, y in zip(*empty_positions):
            moves.append(Action(x, y))
        
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
