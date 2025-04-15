from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple

@dataclass
class Action:
    """表示一个落子动作"""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

class State:
    """表示棋盘状态"""
    def __init__(self, board: Board, current_player: int):
        self.board = board.copy()
        self.current_player = current_player
    
    def clone(self):
        """克隆棋盘状态"""
        return State(self.board, self.current_player)
   
    def zip(self) -> bytes:
        """将状态压缩为字节串
        
        将棋盘状态和当前玩家压缩为字节串。
        棋盘使用 int32 类型，系统字节序。
        玩家编号使用一个字节存储。
        """
        board_array = self.board.astype(np.int32)
        board_bytes = board_array.tobytes()
        player_bytes = self.current_player.to_bytes(1, 'little')
        return board_bytes + player_bytes

    @classmethod
    def from_zipped(cls, zipped: bytes) -> 'State':
        """从字节串恢复状态
        
        从压缩的字节串恢复棋盘状态和当前玩家。
        假设棋盘数据使用 int32 类型存储。
        
        Args:
            zipped: 压缩的字节串，包含棋盘数据(n*n*4字节)和玩家数据(1字节)
        """
        # 计算棋盘大小：总字节数减去玩家标记(1字节)，除以4(int32)，再开平方
        total_board_bytes = len(zipped) - 1
        if total_board_bytes % 4 != 0:
            raise ValueError("无效的字节串长度")
        
        total_cells = total_board_bytes // 4
        board_size = int(np.sqrt(total_cells))
        
        # 验证计算出的大小是否正确
        if board_size * board_size * 4 + 1 != len(zipped):
            raise ValueError("字节串长度与棋盘大小不匹配")
        
        # 解析棋盘数据
        board = np.frombuffer(zipped[:-1], dtype=np.int32).reshape(board_size, board_size)
        current_player = int.from_bytes(zipped[-1:], 'little')
        
        # 验证玩家标记的有效性
        if current_player not in [1, 2]:
            raise ValueError("无效的玩家标记")
        
        return cls(Board(board_size, board), current_player)

    def __hash__(self):
        return hash((self.board.tobytes(), self.current_player))
    
    def __eq__(self, other):
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)

class Board:
    """Hex游戏棋盘"""
    def __init__(
            self, 
            size: int = None, 
            board: Optional[np.ndarray] = None):
        if board is not None:
            self.board = board
            self.size = board.shape[0]
        else:
            self.board = np.zeros((size, size), dtype=int)
            self.size = size

    def reset(self):
        """重置棋盘"""
        self.board.fill(0)
    
    def set_state(self, state: State):
        """设置棋盘状态"""
        assert state.board.shape == self.board.shape, "棋盘大小不匹配"
        self.board = state.board.copy()

    def get_state(self, player_id: int) -> State:
        """获取当前状态"""
        return State(self, player_id)
    
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
    
    def make_move(self, action: Action, player_id: int) -> Tuple[bool, float]:
        """执行一步动作，返回（是否游戏结束，奖励）"""
        assert self.is_valid_move(action)
        
        self.board[action.x, action.y] = player_id
        
        # 检查当前玩家是否获胜
        if self.check_win(player_id):
            return True, 1.0
        
        # 检查是否平局（棋盘已满）
        if len(self.get_valid_moves()) == 0:
            return True, 0.0
        
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
    
    def tobytes(self) -> bytes:
        """将棋盘转换为字节串"""
        return self.board.tobytes()
    
    def tolist(self) -> list:
        """将棋盘转换为列表"""
        return self.board.tolist()
    
    def copy(self) -> 'Board':
        """创建棋盘的深拷贝"""
        new_board = Board(self.size) 
        new_board.board = self.board.copy()
        return new_board

    def paint(
            self, 
            action: Optional[Action] = None, 
            action_probs: Optional[List[float]] = None) -> str:
        """将棋盘状态绘制为 ASCII 字符画
        
        Returns:
            str: ASCII 字符画表示的棋盘，使用:
                ● 表示黑棋(玩家1)
                ○ 表示白棋(玩家2)
                · 表示空位
                X 表示当前动作位置
                (xx%) 表示该位置的动作概率
        """
        size = self.size
        result = []
        cell_width = 8  # 固定每个位置的宽度
        
        # 遍历每一行
        for i in range(size):
            # 添加行首缩进
            row = " " * (i * cell_width // 2)
            
            # 添加每个位置的棋子和概率
            for j in range(size):
                # 获取该位置的概率
                if action_probs is not None:
                    prob = action_probs[i * size + j]
                    prob_str = f"({int(prob * 100)}%)" if prob > 0.01 else " " * self.size
                else:
                    prob_str = " " * self.size
                
                # 如果是当前动作的位置，用 X 标记
                if action is not None and i == action.x and j == action.y:
                    cell = f"X{prob_str}"
                elif self.board[i][j] == 1:
                    cell = f"●{prob_str}"
                elif self.board[i][j] == 2:
                    cell = f"○{prob_str}"
                else:
                    cell = f"·{prob_str}"
                
                # 确保每个位置都是固定宽度
                row += f"{cell:<{cell_width}}"
            row += "\n"
            result.append(row)
        
        return "\n".join(result)