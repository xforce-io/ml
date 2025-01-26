from __future__ import annotations
from abc import abstractmethod
import logging
import random
import time
from typing import Callable, List, Optional, Tuple
from hex.agents.agent import Agent
from hex.hex import Board
import concurrent.futures

from hex.log import DEBUG, ERROR, INFO, WARNING

logger = logging.getLogger(__name__)

class GameResult:
    """游戏结果"""
    def __init__(
            self, 
            agent1: Agent, 
            agent2: Agent, 
            winner: Optional[Agent], 
            moves_count: int,
        ):
        self.agent1 :Agent = agent1
        self.agent2 :Agent = agent2
        self.winner :Agent = winner
        self.moves_count = moves_count

class GameExperiment:
    """游戏实验"""
    def __init__(self):
        self.agent1 = None
        self.agent2 = None
        self.logger = logging.getLogger(__name__)

    def set_agents(self, agent1: Agent, agent2: Agent):
        """设置对弈双方"""
        self.agent1 = agent1
        self.agent2 = agent2
    
    @abstractmethod
    def play_game(self) -> GameResult:
        pass
    
class HexGameExperiment(GameExperiment):
    """游戏实验"""
    def __init__(self, board_size: int = 5):
        super().__init__()
        self.board = Board(board_size)
    
    def play_game(self) -> GameResult:
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
                return GameResult(
                    agent1=self.agent1,
                    agent2=self.agent2,
                    winner=current_agent,
                    moves_count=moves_count
                )
            
            current_agent = self.agent2 if current_agent == self.agent1 else self.agent1
        
        return GameResult(
            agent1=self.agent1,
            agent2=self.agent2,
            winner=None,
            moves_count=moves_count
        )

class ExperimentRunner:
    """实验运行器"""
    def __init__(
            self, 
            total_rounds: int, 
            statistics_rounds: int, 
            num_cores: Optional[int] = None):
        self.total_rounds = total_rounds
        self.statistics_rounds = statistics_rounds
        self.num_cores = num_cores
        # 添加超时设置
        self.timeout = 3600  # 每个批次的超时时间（秒）
    
    def _run_game_batch(
            self, 
            gameExperiment: HexGameExperiment,
            num_games: int) -> List[GameResult]:
        """运行一批游戏并返回结果列表"""
        start_time = time.time()
        results = []
        
        for no in range(num_games):
            if time.time() - start_time > self.timeout:
                WARNING(logger, "Batch timeout reached")
                break
            
            try:
                game_result = gameExperiment.play_game()
                results.append(game_result)
            except Exception as e:
                ERROR(logger, f"Game error: {e}")
                continue
        
        return results

    def run_experiment_in_parallel(
            self, 
            gameExperimentCreator: Callable[[], GameExperiment],
            agent1Creator: Callable[[], Agent],
            agent2Creator: Callable[[], Agent],
            games_per_process: int) -> List[GameResult]:
        experiments = []
        for _ in range(self.num_cores):
            experiment = gameExperimentCreator()
            experiment.set_agents(agent1Creator(), agent2Creator())
            experiments.append(experiment)
        
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            futures = []
            # 提交任务
            for experiment in experiments:
                future = executor.submit(
                    self._run_game_batch,
                    experiment,
                    games_per_process,
                )
                futures.append(future)
            
            # 等待所有任务完成或超时
            try:
                # 使用更长的超时时间
                done, not_done = concurrent.futures.wait(
                    futures,
                    timeout=self.timeout,
                    return_when=concurrent.futures.ALL_COMPLETED
                )
                
                # 处理完成的任务
                for future in done:
                    try:
                        # 收集每个进程的结果
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        ERROR(logger, f"Future error: {e}")
                
                # 处理未完成的任务
                if not_done:
                    WARNING(logger, f"{len(not_done)} tasks did not complete")
                    for future in not_done:
                        future.cancel()
            
            except Exception as e:
                ERROR(logger, f"Batch processing error: {e}")
            
            finally:
                # 确保所有进程都被清理
                executor.shutdown(wait=False)
        
        return all_results

    def run_experiment_and_get_win_rates(
            self, 
            gameExperimentCreator: Callable[[], GameExperiment],
            agent1Creator: Callable[[], Agent],
            agent2Creator: Callable[[], Agent]) -> List[Tuple[float, float]]:
        """并行运行实验"""
        win_rates_history = []
        for round_start in range(0, self.total_rounds, self.statistics_rounds):
            total_wins = [0, 0]

            assert self.statistics_rounds % self.num_cores == 0, "statistics_rounds must be divisible by num_cores"
            games_per_process = self.statistics_rounds // self.num_cores

            start_time = time.time()
            batch_wins = [0, 0]
            agent1 :Agent = None
            agent2 :Agent = None
            
            # 运行实验并获取结果
            results = self.run_experiment_in_parallel(
                gameExperimentCreator,
                agent1Creator,
                agent2Creator,
                games_per_process
            )
            
            # 处理结果
            for game_result in results:
                if game_result.winner:
                    batch_wins[game_result.winner.player_id - 1] += 1
                agent1 = game_result.agent1
                agent2 = game_result.agent2
                    
            # 累加批次结果
            total_wins[0] += batch_wins[0]
            total_wins[1] += batch_wins[1]
            
            # 计算并记录胜率
            total_games = sum(total_wins)
            if total_games > 0:
                win_rate1 = total_wins[0] / total_games
                win_rate2 = total_wins[1] / total_games
                win_rates_history.append((win_rate1, win_rate2))
                
                INFO(logger, f"Round {round_start} - {round_start + self.statistics_rounds}: "
                               f"{agent1.name} win rate = {win_rate1:.2f}, "
                               f"{agent2.name} win rate = {win_rate2:.2f}, "
                               f"cost time = {time.time() - start_time:.2f} seconds")
            
        return win_rates_history
