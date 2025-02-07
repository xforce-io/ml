from __future__ import annotations
from agents.agent import Agent, create_random_agent
from agents.mcts_agent import MCTSAgent, create_mcts_agent
from config import ExperimentConfig, MCTSConfig
from experiment import ExperimentRunner, HexGameExperiment
from rl_basic import GreedyPolicy, RandomPolicy, UCBPolicy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging
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
    runner = ExperimentRunner(
        statistics_rounds=exp_config.statistics_rounds,
        num_cores=exp_config.num_cores
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
    
    def experimentCreator():
        experiment = HexGameExperiment(exp_config.board_size)
        return experiment
    
    results[mcts_config.name] = runner.run_experiment_and_get_win_rates(
        gameExperimentCreator=lambda: experimentCreator(),
        agent1Creator=lambda: agent1,
        agent2Creator=lambda: agent2,
        num_games=exp_config.total_rounds,
        parallel=True
    )
    
    # ExIt智能体实验
    logger.info("\nTraining and evaluating ExIt agent")
    exit_agent = create_exit_agent(
        board_size=exp_config.board_size,
        player_id=2,
        exp_config=exp_config  # 传入实验配置
    )
    
    # 评估ExIt智能体
    agent1 = create_random_agent(player_id=1)
    
    # 打印智能体配置
    log_agent_config(agent1)
    log_agent_config(exit_agent)
    
    results["ExIt-Agent"] = runner.run_experiment_and_get_win_rates(
        gameExperimentCreator=lambda: experimentCreator(),
        agent1Creator=lambda: agent1,
        agent2Creator=lambda: exit_agent,
        num_games=exp_config.total_rounds,
        parallel=True
    )
    
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
        
        experiment = HexGameExperiment(self.board_size)
        # 使用指定的核数
        runner = ExperimentRunner(
            statistics_rounds=100,
            num_cores=self.num_cores
        )
        
        agent1 = create_random_agent(1)
        agent2 = create_mcts_agent(mcts_config, player_id=2)
        experiment.set_agents(agent1, agent2)
        results = runner.run_experiment_and_get_win_rates(
            gameExperimentCreator=lambda: experiment,
            agent1Creator=lambda: agent1,
            agent2Creator=lambda: agent2,
            num_games=400,
            parallel=True
        )
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
    if sys.platform == 'darwin' or sys.platform == 'linux':
        multiprocessing.set_start_method('spawn')
    main() 