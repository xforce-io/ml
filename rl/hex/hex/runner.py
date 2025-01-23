from __future__ import annotations
from agents.agent import Agent, create_random_agent
from agents.mcts_agent import MCTSAgent, create_mcts_agent
from config import ExperimentConfig, MCTSConfig
from experiment import GameExperiment
from rl_basic import GreedyPolicy, LearningConfig, RLAlgorithm, RandomPolicy, UCBPolicy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
import concurrent.futures
import time
import sys
import multiprocessing
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from agents.exit_agent import create_exit_agent

def validate_num_cores(value: str) -> int:
    """验证并返回合法的num_cores值"""
    try:
        num_cores = int(value)
        available_cores = multiprocessing.cpu_count()
        if num_cores < 1:
            raise argparse.ArgumentTypeError(f"num_cores必须大于0，当前值: {num_cores}")
        if num_cores > available_cores:
            logging.warning(f"指定的核心数({num_cores})超过了系统可用核心数({available_cores})，将使用系统最大可用核心数")
            return available_cores
        return num_cores
    except ValueError:
        raise argparse.ArgumentTypeError(f"num_cores必须是整数，当前值: {value}")

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Hex游戏AI训练程序')
    parser.add_argument('--cores', type=validate_num_cores, default=8,
                      help='使用的CPU核心数，默认为8')
    parser.add_argument('--automl', action='store_true',
                      help='是否开启自动参数优化，默认为False')
    return parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExperimentRunner:
    """实验运行器"""
    def __init__(self, total_rounds: int, statistics_rounds: int, num_cores: Optional[int] = None):
        self.total_rounds = total_rounds
        self.statistics_rounds = statistics_rounds
        self.num_cores = num_cores
        self.logger = logging.getLogger(__name__)
        # 添加超时设置
        self.timeout = 600  # 每个批次的超时时间（秒）
    
    def _run_game_batch(self, board_size: int, agent1_config: dict, agent2_config: dict, num_games: int) -> Tuple[int, int]:
        """运行一批游戏并返回胜利统计"""
        wins1 = 0
        wins2 = 0
        
        # 创建新的实验实例
        local_experiment = GameExperiment(board_size)
        
        # 根据配置创建新的智能体
        if agent1_config['type'] == 'Random':
            agent1 = Agent(RandomPolicy(), None, player_id=1, name="Random")
        elif agent1_config['type'] == 'MCTS':
            agent1 = MCTSAgent(
                simulations_per_move=agent1_config['simulations'],
                max_depth=agent1_config['max_depth'],
                c=agent1_config['c'],
                use_rave=agent1_config['use_rave'],
                selection_strategy=agent1_config['strategy'],
                base_rollouts_per_leaf=agent1_config['base_rollouts_per_leaf'],
                player_id=1,
                name=agent1_config['name']
            )
        
        if agent2_config['type'] == 'Random':
            agent2 = Agent(RandomPolicy(), None, player_id=2, name="Random")
        elif agent2_config['type'] == 'MCTS':
            agent2 = MCTSAgent(
                simulations_per_move=agent2_config['simulations'],
                max_depth=agent2_config['max_depth'],
                c=agent2_config['c'],
                use_rave=agent2_config['use_rave'],
                selection_strategy=agent2_config['strategy'],
                base_rollouts_per_leaf=agent2_config['base_rollouts_per_leaf'],
                player_id=2,
                name=agent2_config['name']
            )
        elif agent2_config['type'] == 'DynaQ':
            config = LearningConfig(**agent2_config['learning_config'])
            algorithm = RLAlgorithm(config)
            agent2 = Agent(algorithm.policy, algorithm.estimator, player_id=2, name="DynaQ")
        
        local_experiment.set_agents(agent1, agent2)
        start_time = time.time()
        
        # 添加进度条
        for no in range(num_games):
            if time.time() - start_time > self.timeout:
                self.logger.warning("Batch timeout reached")
                break
            
            try:
                winner, _ = local_experiment.play_game()
                if winner == agent1:
                    wins1 += 1
                elif winner == agent2:
                    wins2 += 1
            except Exception as e:
                self.logger.error(f"Game error: {e}")
                continue
                
        return wins1, wins2
        
    def run_experiment(self, experiment: GameExperiment) -> List[Tuple[float, float]]:
        """并行运行实验"""
        win_rates_history = []
        
        # 提取智能体配置
        agent1_config = self._extract_agent_config(experiment.agent1)
        agent2_config = self._extract_agent_config(experiment.agent2)
        
        for round_start in range(0, self.total_rounds, self.statistics_rounds):
            total_wins = [0, 0]
            games_per_process = self.statistics_rounds // self.num_cores

            start_time = time.time()
            batch_wins = [0, 0]
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                futures = []
                # 提交任务
                for _ in range(self.num_cores):
                    future = executor.submit(
                        self._run_game_batch,
                        experiment.board.size,
                        agent1_config,
                        agent2_config,
                        games_per_process
                    )
                    futures.append(future)
                
                # 等待所有任务完成或超时
                try:
                    # 使用更长的超时时间
                    done, not_done = concurrent.futures.wait(
                        futures,
                        timeout=3600,  # 增加超时时间到60秒
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
                    
                    # 处理完成的任务
                    for future in done:
                        try:
                            wins1, wins2 = future.result(timeout=1)
                            batch_wins[0] += wins1
                            batch_wins[1] += wins2
                        except Exception as e:
                            self.logger.error(f"Future error: {e}")
                    
                    # 处理未完成的任务
                    if not_done:
                        self.logger.warning(f"{len(not_done)} tasks did not complete")
                        for future in not_done:
                            future.cancel()
                
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                
                finally:
                    # 确保所有进程都被清理
                    executor.shutdown(wait=False)
                    
            # 累加批次结果
            total_wins[0] += batch_wins[0]
            total_wins[1] += batch_wins[1]
            
            # 计算并记录胜率
            total_games = sum(total_wins)
            if total_games > 0:
                win_rate1 = total_wins[0] / total_games
                win_rate2 = total_wins[1] / total_games
                win_rates_history.append((win_rate1, win_rate2))
                
                self.logger.info(f"Round {round_start} - {round_start + self.statistics_rounds}: "
                               f"{experiment.agent1.name} win rate = {win_rate1:.2f}, "
                               f"{experiment.agent2.name} win rate = {win_rate2:.2f}, "
                               f"cost time = {time.time() - start_time:.2f} seconds")
            
        return win_rates_history

    def _extract_agent_config(self, agent: Agent) -> dict:
        """提取智能体配置"""
        if isinstance(agent, MCTSAgent):
            return {
                'type': 'MCTS',
                'simulations': agent.policy.simulations_per_move,
                'max_depth': agent.policy.max_depth,
                'c': agent.policy.c,
                'use_rave': agent.policy.use_rave,
                'strategy': agent.policy.selection_strategy,
                'base_rollouts_per_leaf': agent.policy.base_rollouts_per_leaf,
                'name': agent.name
            }
        elif isinstance(agent.policy, RandomPolicy):
            return {
                'type': 'Random',
                'name': agent.name
            }
        elif isinstance(agent.policy, (GreedyPolicy, UCBPolicy)):
            return {
                'type': 'DynaQ',
                'learning_config': {
                    'algorithm_type': 'DynaQ',
                    'initial_learning_rate': agent.estimator.learning_rate,
                    'gamma': agent.estimator.gamma,
                    'initial_epsilon': 0.3,
                    'final_epsilon': 0.05,
                    'planning_steps': 200,
                    'batch_size': 64,
                    'memory_size': 50000
                },
                'name': agent.name
            }

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

def log_agent_config(agent: Agent):
    """打印智能体配置信息"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info(f"Agent Name: {agent.name}")
    
    if isinstance(agent, MCTSAgent):
        logger.info("MCTS Configuration:")
        logger.info(f"- Simulations per move: {agent.policy.simulations_per_move}")
        logger.info(f"- Max depth: {agent.policy.max_depth}")
        logger.info(f"- Exploration constant (c): {agent.policy.c}")
        logger.info(f"- Use RAVE: {agent.policy.use_rave}")
        logger.info(f"- Selection strategy: {agent.policy.selection_strategy}")
        logger.info(f"- Base rollouts per leaf: {agent.policy.base_rollouts_per_leaf}")
    elif isinstance(agent.policy, RandomPolicy):
        logger.info("Random Policy Agent")
    elif isinstance(agent.policy, (GreedyPolicy, UCBPolicy)):
        logger.info("DynaQ Configuration:")
        logger.info(f"- Initial learning rate: {agent.estimator.learning_rate}")
        logger.info(f"- Gamma: {agent.estimator.gamma}")
        if isinstance(agent.policy, GreedyPolicy):
            logger.info(f"- Epsilon: {agent.policy.epsilon}")
        elif isinstance(agent.policy, UCBPolicy):
            logger.info(f"- UCB constant: {agent.policy.c}")
    logger.info("=" * 50)

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载实验配置
    exp_config = ExperimentConfig(num_cores=args.cores)
    
    # 设置日志
    logging.basicConfig(
        level=exp_config.log_level,
        format=exp_config.log_format
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiments with {exp_config.num_cores} cores...")
    
    # 优化MCTS参数（可选）
    optimize_mcts = args.automl  # 从命令行参数获取是否开启自动优化
    if optimize_mcts:
        logger.info("Starting AutoML optimization for MCTS parameters...")
        # 使用配置中指定的核数创建优化器
        optimizer = MCTSOptimizer(
            board_size=exp_config.board_size,
            num_cores=exp_config.num_cores
        )
        best_params, best_score = optimizer.optimize()
        mcts_config = MCTSConfig(**best_params)
        logger.info(f"AutoML optimization completed. Best score: {best_score:.3f}")
    else:
        mcts_config = MCTSConfig()
    
    # 创建实验环境
    experiment = GameExperiment(exp_config.board_size)
    runner = ExperimentRunner(
        exp_config.total_rounds,
        exp_config.statistics_rounds,
        exp_config.num_cores
    )
    
    # 存储实验结果
    results = {}
    
    # MCTS实验
    logger.info("\nRunning experiment with MCTS")
    agent1 = create_random_agent(player_id=1)
    agent2 = create_mcts_agent(mcts_config, player_id=2)
    
    # 打印智能体配置
    log_agent_config(agent1)
    log_agent_config(agent2)
    
    experiment.set_agents(agent1, agent2)
    results[mcts_config.name] = runner.run_experiment(experiment)
    
    # ExIt智能体实验
    logger.info("\nTraining and evaluating ExIt agent")
    exit_agent = create_exit_agent(
        board_size=exp_config.board_size,
        player_id=2,
        exp_config=exp_config  # 传入实验配置
    )
    
    # 评估ExIt智能体
    agent1 = create_random_agent(player_id=1)
    experiment.set_agents(agent1, exit_agent)
    
    # 打印智能体配置
    log_agent_config(agent1)
    log_agent_config(exit_agent)
    
    results["ExIt-Agent"] = runner.run_experiment(experiment)
    
    # 绘制比较图
    plot_comparison(results, exp_config.total_rounds, exp_config.statistics_rounds)

class MCTSOptimizer:
    """MCTS参数优化器"""
    def __init__(self, board_size: int = 5, n_iterations: int = 30, num_cores: int = 8):
        self.board_size = board_size
        self.n_iterations = n_iterations
        self.num_cores = num_cores  # 添加核数参数
        self.X = []
        self.y = []
        
        # 定义参数范围
        self.param_bounds = {
            'simulations': (100, 2000),
            'max_depth': (20, 100),
            'c': (0.5, 3.0),
            'base_rollouts_per_leaf': (5, 40)
        }
        
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
            random_state=42
        )
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """计算采集函数值(使用期望改进)"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # 确保形状正确
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        
        if len(self.y) == 0:
            return mu
        
        # 计算期望改进
        best_f = np.max(self.y)
        imp = mu - best_f
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma!=0)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # 处理可能的数值不稳定性
        ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _sample_parameters(self, num_samples: int = 1000) -> np.ndarray:
        """随机采样参数组合"""
        samples = np.random.rand(num_samples, len(self.param_bounds))
        
        # 将样本映射到参数范围
        for i, (param, (low, high)) in enumerate(self.param_bounds.items()):
            samples[:, i] = samples[:, i] * (high - low) + low
        
        return samples
    
    def _evaluate_mcts(self, params: dict) -> float:
        """评估MCTS配置的性能"""
        mcts_config = MCTSConfig(
            simulations=int(params['simulations']),
            max_depth=int(params['max_depth']),
            c=float(params['c']),
            base_rollouts_per_leaf=int(params['base_rollouts_per_leaf'])
        )
        
        experiment = GameExperiment(self.board_size)
        # 使用指定的核数
        runner = ExperimentRunner(400, 200, num_cores=self.num_cores)
        
        agent1 = create_random_agent(1)
        agent2 = create_mcts_agent(mcts_config, player_id=2)
        experiment.set_agents(agent1, agent2)
        results = runner.run_experiment(experiment)
        win_rates = [rate[1] for rate in results]
        return np.mean(win_rates)
    
    def optimize(self) -> Tuple[dict, float]:
        """运行贝叶斯优化"""
        logger = logging.getLogger(__name__)
        logger.info("Starting MCTS parameter optimization...")
        
        # 初始随机评估
        n_initial = 5
        for _ in range(n_initial):
            params = self._sample_parameters(1)[0]
            param_dict = {
                'simulations': params[0],
                'max_depth': params[1],
                'c': params[2],
                'base_rollouts_per_leaf': params[3]
            }
            score = self._evaluate_mcts(param_dict)
            self.X.append(params)
            self.y.append(score)
        
        # 主优化循环
        for i in range(self.n_iterations - n_initial):
            logger.info(f"Optimization iteration {i+1}/{self.n_iterations-n_initial}")
            
            # 拟合高斯过程
            X_array = np.array(self.X)
            y_array = np.array(self.y)
            self.gp.fit(X_array, y_array)
            
            # 采样新的参数组合
            new_samples = self._sample_parameters(1000)
            ei_values = self._acquisition_function(new_samples)
            
            # 确保ei_values的形状正确
            if len(ei_values) != len(new_samples):
                logger.error(f"Shape mismatch: ei_values shape: {ei_values.shape}, new_samples shape: {new_samples.shape}")
                continue
            
            # 选择最佳参数组合
            best_idx = np.argmax(ei_values)
            best_params = new_samples[best_idx]
            
            # 评估新的参数组合
            param_dict = {
                'simulations': best_params[0],
                'max_depth': best_params[1],
                'c': best_params[2],
                'base_rollouts_per_leaf': best_params[3]
            }
            score = self._evaluate_mcts(param_dict)
            
            # 更新数据
            self.X.append(best_params)
            self.y.append(score)
            
            logger.info(f"Current best score: {max(self.y):.3f}")
            logger.info(f"Parameters: {param_dict}")
        
        # 返回最佳参数组合
        best_idx = np.argmax(self.y)
        best_params = self.X[best_idx]
        best_param_dict = {
            'simulations': int(best_params[0]),
            'max_depth': int(best_params[1]),
            'c': float(best_params[2]),
            'base_rollouts_per_leaf': int(best_params[3])
        }
        
        logger.info("Optimization completed!")
        logger.info(f"Best parameters found: {best_param_dict}")
        logger.info(f"Best score achieved: {self.y[best_idx]:.3f}")
        
        return best_param_dict, self.y[best_idx]

if __name__ == "__main__":
    # 设置多进程启动方法
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn')
    main() 