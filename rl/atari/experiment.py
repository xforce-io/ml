import os
import time
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import gymnasium as gym  # 确保导入gymnasium

from wrappers.atari_wrappers import makeAtari
from algos.base_algo import Algo
from algos.random_algo import AlgoRandom
from algos.dqn_algo import AlgoDQN
from config import Config


class Experiment:
    """实验类，用于设置环境和算法、执行训练和评估"""
    def __init__(self, env_name: str, config: Config = None, render: bool = False):
        """
        初始化实验

        Args:
            env_name (str): Atari 环境名称 (例如 "BreakoutNoFrameskip-v4")
            config (Config, optional): 配置对象，如果不提供则使用默认配置
            render (bool, optional): 是否渲染环境，默认为False
        """
        # 使用提供的配置或默认配置
        self.config = config or Config()
        
        # 保存环境名称，可能与默认配置不同
        self.env_name = env_name
        self.config.general.ENV_NAME = env_name
        
        # 设置随机种子
        self._setSeed(self.config.general.RANDOM_SEED)
        
        # 设置设备
        self.device = torch.device(self.config.general.DEVICE)
        print(f"实验将在设备上运行: {self.device}")
        
        # 保存渲染设置
        self.render = render
        
        # 是否录制视频（默认为False，可通过main.py参数设置）
        self.record_video = False
        
        # 创建环境（仅初始化，不实际构建，因为训练和评估需要单独的环境实例）
        self.env = None
        
        # 当前算法
        self.algo = None
        
        # 训练数据
        self.episode_rewards = []
        
        # 创建模型保存目录
        if not os.path.exists(self.config.general.MODEL_SAVE_DIR):
            os.makedirs(self.config.general.MODEL_SAVE_DIR)
            print(f"创建模型保存目录: {self.config.general.MODEL_SAVE_DIR}")
            
        # 创建视频保存目录
        self.videos_dir = os.path.join(os.getcwd(), "videos")
        if not os.path.exists(self.videos_dir):
            os.makedirs(self.videos_dir)
            print(f"创建视频保存目录: {self.videos_dir}")
            
    def setAlgo(self, algo: Algo):
        """设置当前算法"""
        self.algo = algo

    def _setSeed(self, seed: int):
        """设置随机种子以确保可重复性"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            
    def _createEnv(self, render_mode: Optional[str] = None) -> Any:
        """创建并返回 Atari 环境

        Args:
            render_mode (str, optional): 渲染模式，例如 "human" 或 None

        Returns:
            env: 包装好的 Gymnasium 环境
        """
        # 使用自定义的 Atari 包装函数
        # 训练时无需渲染，评估时可能需要渲染
        env = makeAtari(self.env_name, frame_stack=4, render_mode=render_mode)
            
        print(f"创建环境: {self.env_name}")
        print(f"观察空间形状: {env.observation_space.shape}")
        print(f"动作空间大小: {env.action_space.n}")
        return env
    
    def _runEpisode(self, deterministic: bool = False, 
                    training: bool = False, global_step: int = 0) -> Tuple[float, int, float, int]:
        """运行单个 episode 并返回结果
        
        Args:
            deterministic (bool): 是否使用确定性策略
            training (bool): 是否为训练模式
            global_step (int): 当前的全局步数 (仅在训练时使用)
            
        Returns:
            Tuple[float, int, float, int]: (总奖励, episode长度, 平均损失, 更新后的全局步数)
        """
        episode_reward = 0.0
        episode_length = 0
        episode_loss = 0.0
        done = False
        
        # 重置环境
        state, _ = self.env.reset(seed=self.config.general.RANDOM_SEED + global_step)
        
        while not done:
            # 选择动作
            action = self.algo.selectAction(state, deterministic=deterministic)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # 累加当前 episode 指标
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # 如果是训练模式，则执行学习步骤并更新步数
            if training:
                # 更新算法内部的步数计数器
                global_step += 1
                self.algo.updateStepsDone(global_step)
                
                # 执行学习步骤
                loss = self.algo.learn(state, action, reward, next_state, done)
                if loss is not None:
                    episode_loss += loss
            
            # 更新状态
            state = next_state
        
        # 计算平均损失
        avg_loss = episode_loss / episode_length if episode_length > 0 else 0.0
        
        return episode_reward, episode_length, avg_loss, global_step
    
    def _logTrainingStats(self, step: int, total_steps: int, recent_rewards: deque, 
                          recent_lengths: deque, recent_losses: deque, elapsed_time: float):
        """输出训练统计信息
        
        Args:
            step (int): 当前步数
            total_steps (int): 总步数
            recent_rewards (deque): 最近的奖励队列
            recent_lengths (deque): 最近的长度队列
            recent_losses (deque): 最近的损失队列
            elapsed_time (float): 经过的时间
        """
        steps_per_sec = self.config.general.LOG_INTERVAL / elapsed_time
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0.0
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
        
        print(f"\n步数: {step}/{total_steps} ({100.0 * step / total_steps:.1f}%)")
        print(f"  最近 100 Episode 平均奖励: {avg_reward:.2f}")
        print(f"  最近 100 Episode 平均长度: {avg_length:.1f}")
        print(f"  最近 100 Episode 平均损失: {avg_loss:.4f}")
        
        # 仅当使用 DQN 算法时显示特定信息
        if isinstance(self.algo, AlgoDQN):
            print(f"  当前 Epsilon: {self.algo.getCurrentEpsilon():.4f}")
            print(f"  缓冲区大小: {len(self.algo.memory)}")
        
        print(f"  运行速度: {steps_per_sec:.1f} 步/秒")

    def recordEpisodeVideo(self, step: int) -> None:
        """
        录制一个完整episode的游戏视频
        
        Args:
            step (int): 当前训练步数，用于命名视频文件
        """
        # 创建一个专门用于录制的环境实例，设置render_mode为rgb_array
        video_env = self._createEnv(render_mode="rgb_array")
        
        # 使用gymnasium的录像包装器
        video_env = gym.wrappers.RecordVideo(
            video_env, 
            video_folder=self.videos_dir,
            name_prefix=f"agent_{self.env_name.split('-')[0]}_{step}",
            episode_trigger=lambda x: True  # 录制每个episode
        )
        
        print(f"\n[Step {step}] 开始录制游戏视频...")
        
        # 使用当前算法在视频环境中运行一个episode
        state, _ = video_env.reset(seed=self.config.general.RANDOM_SEED + step)
        done = False
        episode_reward = 0
        episode_length = 0
        
        # 在确定性模式下运行一个完整的episode
        while not done:
            # 由于我们的算法期望的输入格式可能与视频环境输出的不同，可能需要转换
            # 例如，如果算法期望的是张量形式，可能需要转换：
            # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # 但这里我们假设环境封装已经处理好了格式
            
            action = self.algo.selectAction(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = video_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            state = next_state
        
        video_env.close()
        print(f"[Step {step}] 视频录制完成。奖励: {episode_reward:.2f}, 长度: {episode_length}")
        print(f"视频保存在: {self.videos_dir}")
        
    def train(self, num_steps: int = None) -> Dict[str, List]:
        """训练当前算法

        Args:
            num_steps (int, optional): 总训练步数，如不提供则使用配置中的值

        Returns:
            Dict[str, List]: 包含训练指标的字典 (例如 episode 奖励、长度等)
        """
        if self.algo is None:
            raise ValueError("请先设置算法后再开始训练")
            
        # 使用提供的总步数或默认值
        if num_steps is None:
            num_steps = self.config.general.TOTAL_TRAINING_STEPS
            
        # 创建训练环境 (通常无渲染)
        render_mode = "human" if self.render else None
        self.env = self._createEnv(render_mode=render_mode)
        
        print(f"\n开始训练 {self.algo.__class__.__name__} 算法, 总步数: {num_steps}")
        
        # 如果启用了视频录制，显示提示信息
        if self.record_video:
            print(f"视频录制功能已开启。每10000步将录制一个游戏视频，保存在: {self.videos_dir}")
        
        # 训练指标
        train_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_losses': [],
            'steps': [],
            'time': []
        }
        
        # 监控最近的 episode
        recent_rewards = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        
        # 记录开始时间
        start_time = time.time()
        log_time = start_time
        
        # 初始化全局步数和 episode 计数
        global_step = 0
        episode_count = 0
        
        # 训练循环：继续运行，直到达到 total_steps
        while global_step < num_steps:
            # 运行一个 episode
            episode_reward, episode_length, avg_loss, global_step = self._runEpisode(
                deterministic=False,  # 训练模式，使用探索
                training=True,        # 启用学习
                global_step=global_step
            )
            
            # 记录本 episode 指标
            episode_count += 1
            train_metrics['episode_rewards'].append(episode_reward)
            train_metrics['episode_lengths'].append(episode_length)
            train_metrics['episode_losses'].append(avg_loss)
            train_metrics['steps'].append(global_step)
            train_metrics['time'].append(time.time() - start_time)
            
            # 保存奖励用于计算平均值
            self.episode_rewards.append(episode_reward)
            
            # 更新最近的指标队列
            recent_rewards.append(episode_reward)
            recent_lengths.append(episode_length)
            recent_losses.append(avg_loss)
            
            # 打印本 episode 信息
            avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            print(f"Episode {episode_count} - 奖励: {episode_reward:.2f}, 长度: {episode_length}, "
                  f"平均奖励 (100): {avg_recent_reward:.2f}")
            
            # 定期打印训练统计信息 (基于步数)
            if global_step % self.config.general.LOG_INTERVAL < episode_length:
                # 如果当前 episode 跨越了日志间隔点
                now = time.time()
                self._logTrainingStats(
                    step=global_step,
                    total_steps=num_steps,
                    recent_rewards=recent_rewards,
                    recent_lengths=recent_lengths,
                    recent_losses=recent_losses,
                    elapsed_time=now - log_time
                )
                log_time = now
                
            # 每50000步录制一次视频
            if self.record_video and global_step % 50000 < episode_length and global_step >= 10000:
                # 如果当前episode跨越了50000步的倍数点，录制一个视频
                self.recordEpisodeVideo(global_step)
        
        # 最终输出
        self._logTrainingStats(
            step=global_step,
            total_steps=num_steps,
            recent_rewards=recent_rewards,
            recent_lengths=recent_lengths,
            recent_losses=recent_losses,
            elapsed_time=time.time() - log_time
        )
        
        # 训练完成，保存最终模型
        model_path = os.path.join(
            self.config.general.MODEL_SAVE_DIR, 
            f"{self.algo.__class__.__name__.lower()}_{self.env_name}_final.pth"
        )
        self.algo.save(self.config.general.MODEL_SAVE_DIR, f"{self.algo.__class__.__name__.lower()}_{self.env_name}_final.pth")
        
        print("\n训练完成!")
        
        return train_metrics

    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """评估当前算法

        Args:
            num_episodes (int, optional): 要评估的 episodes 数量

        Returns:
            Dict[str, float]: 包含评估指标的字典
        """
        if self.algo is None:
            raise ValueError("请先设置算法后再开始评估")
            
        # 使用提供的 episodes 数量或默认值
        if num_episodes is None:
            num_episodes = self.config.general.EVAL_EPISODES
        
        # 创建评估环境 (可能启用渲染)
        render_mode = "human" if self.render else None
        self.env = self._createEnv(render_mode=render_mode)
        
        # 切换算法到评估模式
        self.algo.setEvalMode()
        
        print(f"\n开始评估 {self.algo.__class__.__name__} 算法, {num_episodes} 个 episodes")
        
        # 评估指标
        total_reward = 0.0
        total_length = 0
        victories = 0  # 可选：在某些游戏中获胜的次数
        
        # 评估循环
        for ep in range(num_episodes):
            # 运行一个 episode
            episode_reward, episode_length, _, _ = self._runEpisode(
                deterministic=True,  # 评估模式，使用确定性策略
                training=False       # 不启用学习
            )
            
            # 累加指标
            total_reward += episode_reward
            total_length += episode_length
            
            # 可选：判断是否获胜
            if episode_reward > self.config.general.VICTORY_THRESHOLD:
                victories += 1
                
            # 打印结果
            print(f"Episode {ep+1}/{num_episodes} - 奖励: {episode_reward:.2f}, 长度: {episode_length}")
        
        # 计算平均值
        avg_reward = total_reward / num_episodes
        avg_length = total_length / num_episodes
        win_rate = victories / num_episodes
        
        # 打印总结
        print(f"\n评估结果 ({num_episodes} episodes):")
        print(f"  平均奖励: {avg_reward:.2f}")
        print(f"  平均长度: {avg_length:.1f}")
        print(f"  胜率: {win_rate:.2f} ({victories}/{num_episodes})")
        
        # 返回指标
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'win_rate': win_rate
        } 