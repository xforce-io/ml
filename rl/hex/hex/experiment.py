from __future__ import annotations
from abc import abstractmethod
import logging
import random
import time
import traceback
from typing import Callable, List, Optional, Tuple
from hex.agents.agent import Agent, GameResult
from hex.hex import Board
import concurrent.futures
import multiprocessing

from hex.log import DEBUG, ERROR, INFO, WARNING

logger = logging.getLogger(__name__)

class GameExperiment:
    """游戏实验"""
    def __init__(self):
        self.agent1 = None
        self.agent2 = None

    def set_agents(self, agent1: Agent, agent2: Agent):
        """设置对弈双方"""
        self.agent1 = agent1
        self.agent2 = agent2
    
    @abstractmethod
    def play_game(self) -> GameResult:
        pass
    
class HexGameExperiment(GameExperiment):
    """游戏实验"""
    def __init__(self, board_size: int):
        super().__init__()
        self.board = Board(board_size)
    
    def play_game(self, cold_start: bool) -> GameResult:
        """进行一局游戏，返获胜步数"""
        t0 = time.time()

        start_agent = random.choice([self.agent1, self.agent2])
        current_agent = start_agent
        self.board.reset()
        moves_count = 0
        MAX_MOVES = 100

        other_player = lambda agent: self.agent1 if agent == self.agent2 else self.agent2
        
        while moves_count < MAX_MOVES:
            action = current_agent.choose_action(self.board, cold_start)
            game_over, reward = self.board.make_move(action, current_agent.player_id)
            moves_count += 1

            if game_over:
                INFO(logger, f"Game over, start_agent = {start_agent.player_id}, winner_agent = {current_agent.player_id}, reward_of_agent1 = {1 if current_agent == self.agent1 else -1}, cost time = {time.time() - t0:.2f} seconds")
                if current_agent == self.agent1:
                    experiences = self.agent1.reward(1, self.board)
                    self.agent2.reward(-1, self.board)
                else:
                    experiences = self.agent1.reward(-1, self.board)
                    self.agent2.reward(1, self.board)
                return GameResult(
                    agent1_id=self.agent1.player_id,
                    agent2_id=self.agent2.player_id,
                    first_agent_id=start_agent.player_id,
                    winner_id=current_agent.player_id,
                    experiences=experiences,
                    moves_count=moves_count
                )
            
            current_agent = other_player(current_agent)
        
        experiences = current_agent.reward(0, self.board)
        other_player(current_agent).reward(0, self.board)
        INFO(logger, f"Game over, no winner, cost time = {time.time() - t0:.2f} seconds")
        return GameResult(
            agent1_id=self.agent1.player_id,
            agent2_id=self.agent2.player_id,
            first_agent_id=start_agent.player_id,
            winner_id=None,
            experiences=experiences,
            moves_count=moves_count
        )

class ExperimentRunner:
    """实验运行器"""
    def __init__(
            self, 
            statistics_rounds: int, 
            num_cores: Optional[int] = None):
        self.statistics_rounds = statistics_rounds
        self.num_cores = num_cores or multiprocessing.cpu_count()
        self.timeout = 3600  # 每个批次的超时时间（秒）

    def run_experiments(
            self, 
            gameExperimentCreator: Callable[[], GameExperiment],
            agent1Creator: Callable[[], Agent],
            agent2Creator: Callable[[], Agent],
            num_games: int,
            cold_start: bool=False,
            parallel: bool=True) -> List[GameResult]:
        """运行实验"""
        t0 = time.time()
        
        if not parallel:
            # 单进程执行的代码保持不变
            experiment = gameExperimentCreator()
            experiment.set_agents(agent1Creator(), agent2Creator())
            results = self._run_game_batch(experiment, num_games, self.timeout, cold_start)
            INFO(logger, f"Run {num_games} games in single process, costSec[{time.time() - t0:.2f}]")
            return results

        assert num_games % self.num_cores == 0, "num_games must be divisible by num_cores"
        games_per_process = num_games // self.num_cores

        # 修改这部分代码，将对象创建移到子进程中
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            futures = []
            # 提交任务
            for _ in range(self.num_cores):
                future = executor.submit(
                    ExperimentRunner._run_batch, 
                    gameExperimentCreator, 
                    agent1Creator, 
                    agent2Creator, 
                    games_per_process, 
                    self.timeout,
                    cold_start)
                futures.append(future)
            
            # 等待所有任务完成或超时
            try:
                done, not_done = concurrent.futures.wait(
                    futures,
                    timeout=self.timeout,
                    return_when=concurrent.futures.ALL_COMPLETED
                )
                
                # 处理完成的任务
                for future in done:
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        ERROR(logger, f"Future error: {e} traceback: {traceback.format_exc()}")
                
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
        
        INFO(logger, f"Run {num_games} games in parallel, costSec[{time.time() - t0:.2f}]")
        return all_results

    def run_experiment_and_get_win_rates(
            self, 
            gameExperimentCreator: Callable[[], GameExperiment],
            agent1Creator: Callable[[], Agent],
            agent2Creator: Callable[[], Agent],
            num_games: int = 1000,
            cold_start: bool = False,
            parallel: bool = True) -> List[Tuple[float, float]]:
        """并行运行实验"""
        win_rates_history = []
        for round_start in range(0, num_games, self.statistics_rounds):
            total_wins = [0, 0]

            start_time = time.time()
            batch_wins = [0, 0]
            
            # 运行实验并获取结果
            results = self.run_experiments(
                gameExperimentCreator,
                agent1Creator,
                agent2Creator,
                num_games=self.statistics_rounds,
                cold_start=cold_start,
                parallel=parallel
            )
            
            # 处理结果
            for game_result in results:
                if game_result.winner_id is not None:
                    batch_wins[game_result.winner_id - 1] += 1
                agent1_id = game_result.agent1_id
                agent2_id = game_result.agent2_id
                    
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
                      f"{agent1_id} win rate = {win_rate1:.2f}, "
                      f"{agent2_id} win rate = {win_rate2:.2f}, "
                      f"cost time = {time.time() - start_time:.2f} seconds")
            
        return win_rates_history

    @staticmethod
    def _run_batch(
            gameExperimentCreator: Callable[[], GameExperiment],
            agent1Creator: Callable[[], Agent],
            agent2Creator: Callable[[], Agent],
            games_per_process: int,
            timeout: int,
            cold_start: bool) -> List[GameResult]:
        logger = logging.getLogger(__name__)
        try:
            experiment = gameExperimentCreator()
            experiment.set_agents(agent1Creator(), agent2Creator())
            return ExperimentRunner._run_game_batch(
                experiment, 
                games_per_process, 
                timeout,
                cold_start)
        except Exception as e:
            ERROR(logger, f"Process error: {e} traceback: {traceback.format_exc()}")
            return []

    @staticmethod
    def _run_game_batch(
            gameExperiment: HexGameExperiment,
            num_games: int,
            timeout: int,
            cold_start: bool) -> List[GameResult]:
        """运行一批游戏并返回结果列表"""
        logger = logging.getLogger(__name__)
        start_time = time.time()
        results = []
        
        for no in range(num_games):
            if time.time() - start_time > timeout:
                WARNING(logger, "Batch timeout reached")
                break
            
            try:
                game_result = gameExperiment.play_game(cold_start)
                results.append(game_result)
            except Exception as e:
                ERROR(logger, f"Game error: {e} traceback: {traceback.format_exc()}")
                continue
        
        INFO(logger, f"Batch game finished, cost time = {time.time() - start_time:.2f} seconds")
        return results

