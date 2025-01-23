from __future__ import annotations
import logging
import random
from typing import Tuple
from hex.agents.agent import Agent
from hex.hex import Board

class GameExperiment:
    """游戏实验"""
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

