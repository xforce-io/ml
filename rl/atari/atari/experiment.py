import logging
from atari.log import INFO
import os
import time
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import gymnasium as gym  # 确保导入gymnasium

from atari.wrappers.atari_wrappers import makeAtari
from atari.algos.base_algo import Algo
from atari.algos.dqn_algo import AlgoDQN
from atari.config import Config

logger = logging.getLogger(__name__)

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
        INFO(logger, f"实验将在设备上运行: {self.device}")
        
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
        self.model_save_dir = self.config.general.MODEL_SAVE_DIR
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            INFO(logger, f"创建模型保存目录: {self.model_save_dir}")
            
        # 创建视频保存目录
        self.video_save_dir = self.config.general.VIDEO_SAVE_DIR
        if not os.path.exists(self.video_save_dir):
            os.makedirs(self.video_save_dir)
            INFO(logger, f"创建视频保存目录: {self.video_save_dir}")
            
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
        # 判断是否是Mario环境
        if 'SuperMarioBros' in self.env_name:
            # 提取世界和关卡信息，例如从"SuperMarioBros-1-1-v0"提取"1-1"
            world_stage = None
            if '-' in self.env_name and 'v' in self.env_name:
                parts = self.env_name.split('-')
                if len(parts) >= 3:  # 例如：SuperMarioBros-1-1-v0
                    world_stage = f"{parts[1]}-{parts[2].split('v')[0]}"
            
            # 使用makeMario函数创建环境
            from atari.wrappers.mario_wrappers import makeMario
            env = makeMario(
                worldStage=world_stage,
                maxEpisodeSteps=10000,
                frameSkip=4,
                grayscale=True,
                resizeShape=(84, 84),
                frameStack=4,
                normalizeObs=True,
                renderMode=render_mode
            )
            # MarioEnv 已经内置了时间限制逻辑，不需要额外的 TimeLimit 包装器
        else:
            # 使用makeAtari函数创建Atari环境
            from atari.wrappers.atari_wrappers import makeAtari
            env = makeAtari(
                self.env_name, 
                frameStack=4, 
                renderMode=render_mode
            )
            
        INFO(logger, f"创建环境: {self.env_name}")
        INFO(logger, f"观察空间形状: {env.observation_space.shape}")
        INFO(logger, f"动作空间大小: {env.action_space.n}")
        return env
    
    def _runEpisode(self, deterministic: bool = False, 
                    training: bool = False, global_step: int = 0) -> Tuple[float, int, float, int]:
        """运行单个 episode 并返回结果"""
        episode_reward = 0.0
        episode_length = 0
        episode_loss = 0.0
        
        # 重置环境
        seed = self.config.general.RANDOM_SEED + global_step
        try:
            # 尝试使用带有seed参数的reset方法
            state, _ = self.env.reset(seed=seed)
        except (TypeError, ValueError):
            # 如果环境不支持seed参数，则直接重置
            reset_result = self.env.reset()
            # 处理不同环境可能返回的不同格式
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                state, _ = reset_result
            else:
                state = reset_result
        
        # 性能统计
        total_env_time = 0
        total_action_time = 0
        total_learn_time = 0
        episode_start = time.time()
        
        done = False
        
        while not done:
            # 选择动作
            t0 = time.time()
            action = self.algo.selectAction(state, deterministic=deterministic)
            total_action_time += time.time() - t0
            
            # 执行动作
            t0 = time.time()
            # 直接期望 Gymnasium API 的5个返回值
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            total_env_time += time.time() - t0
            
            # 累加当前 episode 指标
            episode_reward += reward
            episode_length += 1
            
            # 如果是训练模式，则执行学习步骤并更新步数
            if training:
                # 更新算法内部的步数计数器
                global_step += 1
                self.algo.updateStepsDone(global_step)
                
                # 执行学习步骤
                t0 = time.time()
                loss = self.algo.learn(state, action, reward, next_state, done)
                total_learn_time += time.time() - t0
                if loss is not None:
                    episode_loss += loss
            
            # 更新状态
            state = next_state

        # 每1000步输出一次性能统计
        if global_step % 1000 < episode_length:
            total_time = time.time() - episode_start
            INFO(logger, f"\n性能统计:")
            INFO(logger, f"环境执行时间占比: {total_env_time/total_time*100:.1f}%")
            INFO(logger, f"动作选择时间占比: {total_action_time/total_time*100:.1f}%")
            INFO(logger, f"学习时间占比: {total_learn_time/total_time*100:.1f}%")
            INFO(logger, f"每步平均耗时: {total_time/episode_length*1000:.1f}ms")
        
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
        
        INFO(logger, f"\n步数: {step}/{total_steps} ({100.0 * step / total_steps:.1f}%)")
        INFO(logger, f"  最近 100 Episode 平均奖励: {avg_reward:.2f}")
        INFO(logger, f"  最近 100 Episode 平均长度: {avg_length:.1f}")
        INFO(logger, f"  最近 100 Episode 平均损失: {avg_loss:.4f}")
        
        # 仅当使用 DQN 算法时显示特定信息
        if isinstance(self.algo, AlgoDQN):
            INFO(logger, f"  当前 Epsilon: {self.algo.getCurrentEpsilon():.4f}")
            INFO(logger, f"  缓冲区大小: {len(self.algo.memory)}")
        
        INFO(logger, f"  运行速度: {steps_per_sec:.1f} 步/秒")

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
            video_folder=self.video_save_dir,
            name_prefix=f"agent_{self.env_name.split('-')[0]}_{step}",
            episode_trigger=lambda x: True  # 录制每个episode
        )
        
        INFO(logger, f"\n[Step {step}] 开始录制游戏视频...")
        
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
        INFO(logger, f"[Step {step}] 视频录制完成。奖励: {episode_reward:.2f}, 长度: {episode_length}")
        INFO(logger, f"视频保存在: {self.video_save_dir}")
        
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
        
        INFO(logger, f"\n开始训练 {self.algo.__class__.__name__} 算法, 总步数: {num_steps}")
        
        # 如果启用了视频录制，显示提示信息
        if self.record_video:
            INFO(logger, f"视频录制功能已开启。每{self.config.general.VIDEO_SAVE_INTERVAL}步将录制一个游戏视频，保存在: {self.video_save_dir}")
        
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
            INFO(logger, f"Episode {episode_count} - 奖励: {episode_reward:.2f}, 长度: {episode_length}, "
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
                
            if self.record_video and global_step >= 10000:
                if global_step % self.config.general.VIDEO_SAVE_INTERVAL < episode_length:
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
        
        INFO(logger, "\n训练完成!")
        
        return train_metrics

    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """评估当前算法
        
        Args:
            num_episodes (int, optional): 要评估的Episode数量，如不提供则使用配置中的值
            
        Returns:
            Dict[str, float]: 包含评估指标的字典 (例如平均奖励、标准差等)
        """
        if self.algo is None:
            raise ValueError("请先设置算法后再开始评估")
        
        num_episodes = num_episodes or self.config.general.EVAL_EPISODES
        self.algo.setEvalMode()  # 设置为评估模式 (例如关闭探索)
        
        # 如果需要渲染，创建一个用于可视化的单独环境
        render_mode = "human" if self.render else None
        
        # 保留当前的训练环境引用
        train_env = self.env
        
        # 若为Atari游戏使用_createEnv方法创建环境，若为Mario游戏则沿用main.py中创建的环境
        if "SuperMarioBros" in self.env_name:
            # 对于Mario游戏，环境已经在main.py中创建，这里仅在需要重置时才重新创建
            if render_mode == "human" and (not hasattr(self.env.env, 'render') or not hasattr(self.env.env, 'reset')):
                from atari.wrappers import makeMario
                world_stage = None
                if "-" in self.env_name:
                    parts = self.env_name.split("-")
                    if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                        world_stage = f"{parts[1]}-{parts[2]}"
                self.env = makeMario(worldStage=world_stage, renderMode=render_mode)
        else:
            # 对于Atari游戏，使用原有的环境创建方法
            self.env = self._createEnv(render_mode=render_mode)
        
        INFO(logger, f"\n开始评估 {self.env_name} 环境中的 {self.algo.__class__.__name__}")
        INFO(logger, f"评估 episodes: {num_episodes}, 确定性策略: {self.render}")
        
        episode_rewards = []
        episode_lengths = []
        
        for i in range(num_episodes):
            # 运行单个evaluation episode
            episode_reward, episode_length, _, _ = self._runEpisode(deterministic=True)
            
            # 累积奖励和长度
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            INFO(logger, f"  Episode {i+1}/{num_episodes}: 奖励 = {episode_reward:.2f}, 长度 = {episode_length}")
        
        # 计算统计信息
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        INFO(logger, f"\n评估结果: ")
        INFO(logger, f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        INFO(logger, f"  平均长度: {mean_length:.1f}")
        
        # 判断是否达到"胜利"标准
        if mean_reward >= self.config.general.VICTORY_THRESHOLD:
            INFO(logger, f"  ***胜利!*** 平均奖励超过阈值 {self.config.general.VICTORY_THRESHOLD:.1f}")
        
        # 恢复训练环境
        self.env = train_env
        self.algo.setTrainMode()  # 恢复训练模式
        
        # 返回评估结果
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "rewards": episode_rewards,
            "lengths": episode_lengths
        } 