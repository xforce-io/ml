import logging
from atari.log import ERROR, INFO, WARNING
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import time

from .base_algo import Algo
from atari.models.dqn_model import DQN             # 使用绝对导入
from atari.utils.replay_buffer import ReplayBuffer # 使用绝对导入
from atari.utils.prioritized_replay_buffer import PrioritizedReplayBuffer # 使用绝对导入

logger = logging.getLogger(__name__)

class AlgoDQN(Algo):
    """DQN 算法实现"""

    DEBUG_INTERVAL = 1000
    
    def __init__(self, env, config, device):
        super().__init__(env, config, device)
        INFO(logger, f"初始化 DQN 算法 (AlgoDQN) 在设备: {self.device}")

        # 处理配置，支持直接传入DQNConfig或整个Config对象
        if hasattr(config, 'BATCH_SIZE'):
            # 直接使用传入的DQNConfig
            dqn_config = config
        elif hasattr(config, 'dqn'):
            # 使用Config.dqn
            dqn_config = config.dqn
        else:
            raise ValueError("无法找到DQN配置参数")

        # 从配置中获取 DQN 特定参数
        self.batch_size = dqn_config.BATCH_SIZE
        self.gamma = dqn_config.GAMMA
        self.target_update_frequency = dqn_config.TARGET_UPDATE_FREQUENCY
        self.epsilon_start = dqn_config.EPSILON_START
        self.epsilon_end = dqn_config.EPSILON_END
        self.epsilon_decay_steps = dqn_config.EPSILON_DECAY_STEPS
        self.learning_starts = dqn_config.LEARNING_STARTS
        self.grad_clip_value = dqn_config.GRAD_CLIP_VALUE
        
        # 获取优先经验回放相关参数
        self.use_prioritized_replay = dqn_config.USE_PRIORITIZED_REPLAY if hasattr(dqn_config, 'USE_PRIORITIZED_REPLAY') else False
        if self.use_prioritized_replay:
            self.per_alpha = dqn_config.PER_ALPHA
            self.per_beta_start = dqn_config.PER_BETA_START
            self.per_beta_increment = dqn_config.PER_BETA_INCREMENT
            self.per_epsilon = dqn_config.PER_EPSILON
            INFO(logger, f"启用优先经验回放 (PER) - alpha: {self.per_alpha}, beta_start: {self.per_beta_start}")

        # 检测游戏类型
        if hasattr(config.general, 'ENV_NAME') and 'SuperMarioBros' in config.general.ENV_NAME:
            game_type = "mario"
            INFO(logger, f"检测到 Mario 游戏环境，使用适合的卷积架构")
        else:
            game_type = "atari"
            INFO(logger, f"使用标准 Atari 游戏卷积架构")

        # 创建策略网络和目标网络
        self.policy_net = DQN(self.input_shape, self.num_actions, game_type=game_type).to(self.device)
        self.target_net = DQN(self.input_shape, self.num_actions, game_type=game_type).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=dqn_config.LEARNING_RATE)

        # 创建经验回放缓冲区
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                dqn_config.REPLAY_BUFFER_CAPACITY, 
                device=self.device,
                alpha=self.per_alpha,
                beta=self.per_beta_start,
                beta_increment=self.per_beta_increment,
                epsilon=self.per_epsilon
            )
        else:
            self.memory = ReplayBuffer(dqn_config.REPLAY_BUFFER_CAPACITY, device=self.device)

        # 评估时的固定 epsilon
        self.eval_epsilon = 0.001  # 可以从配置中获取如果有的话
        
        # 最后一次目标网络更新的步数
        self.last_target_update = 0
        
        # 跟踪训练统计信息
        self.recent_losses = []

    def _calculateEpsilon(self):
        """计算当前的 epsilon 值"""
        if self.steps_done >= self.epsilon_decay_steps:
            return self.epsilon_end
        else:
            fraction = min(1.0, self.steps_done / self.epsilon_decay_steps)
            return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def selectAction(self, state, deterministic=False):
        """根据当前状态和 epsilon-greedy 策略选择动作"""
        if deterministic:
            # 评估模式：使用极小的探索率
            epsilon = self.eval_epsilon
        else:
            # 训练模式：使用衰减的 epsilon
            epsilon = self._calculateEpsilon()

        # epsilon-greedy
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)

        return action

    def _computeGradNorm(self):
        """计算梯度范数，用于调试"""
        total_norm = 0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def learn(self, state, action, reward, next_state, done):
        """存储经验，并在满足条件时执行学习步骤"""
        # 1. 存储经验
        # 注意转换类型以匹配 ReplayBuffer 的 push 签名
        t0 = time.time()
        self.memory.push(state, np.array([action]), np.array([reward]), next_state, np.array([done]))
        buffer_time = time.time() - t0

        # 2. 检查是否可以开始学习
        # 条件1: 交互步数是否达到 'learning_starts' (配置参数)
        # 条件2: 经验回放缓冲区中的样本数量是否足够一个批次 (batch_size)
        if self.steps_done < self.learning_starts:
            if self.steps_done % self.DEBUG_INTERVAL == 0:
                INFO(logger, f"[等待学习] 步数: {self.steps_done}/{self.learning_starts}, 经验缓冲区: {len(self.memory)}/{self.batch_size}")
            return None # 如果不满足条件，则暂时不学习，直接返回
            
        if len(self.memory) < self.batch_size:
            if self.steps_done % self.DEBUG_INTERVAL == 0:
                INFO(logger, f"[缓冲区填充中] 经验缓冲区: {len(self.memory)}/{self.batch_size}")
            return None # 样本不足，不学习

        # --- 如果满足学习条件，则执行以下步骤 ---

        # 3. 从缓冲区采样
        t0 = time.time()
        if self.use_prioritized_replay:
            # 对于优先经验回放，sample会返回额外的索引和权重
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            # 检查权重是否合理
            if torch.any(weights.isnan()) or torch.any(weights.isinf()):
                WARNING(logger, "[警告] 权重中包含 nan 或 inf 值")
                weights = torch.clamp(weights, min=1e-5, max=1e5)  # 限制权重范围
        else:
            # 普通经验回放
            batch = self.memory.sample(self.batch_size)
            if batch is None: # 以防万一采样失败
                WARNING(logger, "[警告] 采样失败")
                return None
            # 解包批次数据
            states, actions, rewards, next_states, dones = batch
            # 创建虚拟权重 (全1) 用于保持代码一致性
            weights = torch.ones((self.batch_size, 1), device=self.device)
            
        buffer_time += time.time() - t0

        # 4. 计算 Q(s_t, a_t) - 当前状态-动作对的 Q 值 (预测值)
        t0 = time.time()
        self.policy_net.train() # 确保策略网络处于训练模式（启用 dropout, batchnorm 等）
        # 将采样到的 'states' 输入策略网络，得到这些状态下所有可能动作的 Q 值
        q_values = self.policy_net(states)
        # 检查 q_values 是否包含 nan 或 inf
        if torch.any(q_values.isnan()) or torch.any(q_values.isinf()):
            WARNING(logger, "[警告] q_values 中包含 nan 或 inf 值")
        # 从 q_values 中，根据采样到的 'actions'，精确地提取出实际执行动作对应的 Q 值
        q_values_for_actions = q_values.gather(1, actions)

        # 5. 计算 V(s_{t+1}) - 下一个状态的价值 (用于构建目标值, 使用 Double DQN)
        with torch.no_grad(): # 不需要计算梯度
            self.target_net.eval() # 确保目标网络处于评估模式
            # 切换策略网络到评估模式以选择动作 (不影响梯度计算，因为在 no_grad 块内)
            self.policy_net.eval() 

            # --- Double DQN 核心逻辑 ---
            # 步骤 1: 使用当前策略网络 (policy_net) 找出下一状态 (next_states) 中 Q 值最大的动作
            # .detach() 不是必须的，因为在 no_grad() 中，但加上无害
            best_next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            
            # 可选: 检查动作索引的形状是否正确
            if best_next_actions.shape[0] != next_states.shape[0]:
                WARNING(logger, f"Double DQN: 策略网络选择的动作数量 ({best_next_actions.shape[0]}) 与批次大小 ({next_states.shape[0]}) 不符")

            # 步骤 2: 使用目标网络 (target_net) 获取下一状态的所有 Q 值
            next_q_values_target = self.target_net(next_states)
            # 可选: 检查目标网络输出
            if torch.any(next_q_values_target.isnan()) or torch.any(next_q_values_target.isinf()):
                WARNING(logger, "Double DQN: 目标网络输出的 next_q_values 中包含 nan 或 inf 值")
                
            # 步骤 3: 使用目标网络 (target_net) 评估由策略网络选出的最佳动作 (best_next_actions) 的 Q 值
            # 使用 gather 从目标网络的输出中，根据策略网络选出的动作索引，提取 Q 值
            selected_next_q_values = next_q_values_target.gather(1, best_next_actions)
            # 可选: 检查选定的 Q 值
            if torch.any(selected_next_q_values.isnan()) or torch.any(selected_next_q_values.isinf()):
                WARNING(logger, "Double DQN: 选定的下一状态 Q 值包含 nan 或 inf 值")
            # --- Double DQN 核心逻辑结束 ---

            # 计算目标 Q 值 (TD Target): R + γ * Q_target(s', argmax_a Q_policy(s', a))
            # (1 - dones) 的作用是：如果一个状态是终止状态 (done=True)，那么其后续价值为 0
            target_q_values = rewards + (self.gamma * selected_next_q_values * (1 - dones))
            # 检查最终的目标 Q 值
            if torch.any(target_q_values.isnan()) or torch.any(target_q_values.isinf()):
                WARNING(logger, "Double DQN: 计算得到的 target_q_values 中包含 nan 或 inf 值")
        
        # 将策略网络恢复到训练模式（因为在 with no_grad() 之前设置了 train()，
        # 并且 loss.backward() 需要它处于训练模式）
        self.policy_net.train() 
        network_time = time.time() - t0

        # 6. 计算TD误差和损失 (Loss)
        t0 = time.time()
        # 计算TD误差 (用于更新优先级)
        # 注意: 这里的 q_values_for_actions 仍然是用 policy_net 在训练模式下计算的
        td_errors = target_q_values - q_values_for_actions 
        # 检查 td_errors 是否包含 nan 或 inf
        if torch.any(td_errors.isnan()) or torch.any(td_errors.isinf()):
            WARNING(logger, "td_errors 中包含 nan 或 inf 值")
        
        # 计算带权重的损失
        # 使用smooth_l1_loss (Huber Loss)，对异常值不那么敏感
        # 注意我们将每个样本的损失乘以对应的重要性采样权重
        elementwise_loss = F.smooth_l1_loss(q_values_for_actions, target_q_values, reduction='none')
        loss = (elementwise_loss * weights).mean()
        # 检查 loss 是否为 nan 或 inf
        if torch.isnan(loss) or torch.isinf(loss):
            WARNING(logger, "loss 为 nan 或 inf")

        # 7. 优化模型 (执行反向传播和参数更新)
        self.optimizer.zero_grad() # 清除上一轮的梯度
        loss.backward() # 计算损失相对于策略网络参数的梯度
        if self.grad_clip_value is not None: # 如果设置了梯度裁剪值
            # 对策略网络的梯度进行裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip_value)
        self.optimizer.step() # 使用优化器 (例如 Adam) 根据梯度更新策略网络的参数
        optimizer_time = time.time() - t0

        # 8. 如果使用优先经验回放，更新样本优先级
        if self.use_prioritized_replay:
            # 使用最新计算的TD误差更新优先级
            self.memory.update_priorities(indices, td_errors.detach().abs())

        # 9. 定期更新目标网络
        # 检查当前总步数距离上次更新目标网络是否超过了指定频率 (target_update_frequency)
        if self.steps_done - self.last_target_update >= self.target_update_frequency:
            INFO(logger, f"\n[Step {self.steps_done}] 更新目标网络")
            # 将策略网络的最新权重复制给目标网络
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # 更新上次更新的步数记录
            self.last_target_update = self.steps_done
            
        # 10. 记录损失值用于统计
        loss_value = loss.item()
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)
            
        # 11. 添加定期调试信息
        if self.steps_done % self.DEBUG_INTERVAL == 0:
            INFO(logger, f"\n[DQN性能分析 Step {self.steps_done}]")
            INFO(logger, f"  - 缓冲区操作时间: {buffer_time*1000:.1f}ms")
            INFO(logger, f"  - 网络计算时间: {network_time*1000:.1f}ms")
            INFO(logger, f"  - 优化器时间: {optimizer_time*1000:.1f}ms")
            INFO(logger, f"  - 总时间: {(buffer_time + network_time + optimizer_time)*1000:.1f}ms")
            
            # 其他调试信息
            avg_reward = np.mean(rewards.cpu().numpy())
            avg_max_q = np.mean(q_values.max(dim=1)[0].detach().cpu().numpy())
            avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0
            INFO(logger, f"  - 平均奖励: {avg_reward:.4f}")
            INFO(logger, f"  - 平均最大Q值: {avg_max_q:.4f}")
            INFO(logger, f"  - 平均损失: {avg_loss:.6f}")
            INFO(logger, f"  - 当前Epsilon: {self._calculateEpsilon():.4f}")
            INFO(logger, f"  - 缓冲区大小: {len(self.memory)}")
            INFO(logger, f"  - 批次大小: {states.shape[0]}")
            
            if self.use_prioritized_replay:
                INFO(logger, f"  - PER Beta值: {self.memory.beta:.4f}")
                
        return loss_value

    def visualizeQDistribution(self, num_samples=10):
        """可视化Q值分布来检查学习进展"""
        if len(self.memory) < num_samples:
            WARNING(logger, "经验回放缓冲区样本不足")
            return
        
        # 从缓冲区获取样本
        indices = np.random.choice(len(self.memory), num_samples, replace=False)
        states_np = [] # 用于收集 NumPy 格式的状态
        for idx in indices:
            # 假设状态存储在内存元组的第一个位置
            state_data = self.memory.memory[idx][0]
            
            # 确保状态数据在 CPU 上并且是 NumPy 格式
            if isinstance(state_data, torch.Tensor):
                # 如果是张量，移动到 CPU 并转为 NumPy
                states_np.append(state_data.cpu().numpy())
            elif isinstance(state_data, np.ndarray):
                # 如果已经是 NumPy 数组，直接添加
                states_np.append(state_data)
            else:
                # 尝试将其他类型转换为 NumPy
                try:
                    states_np.append(np.array(state_data))
                except Exception as e:
                    WARNING(logger, f"无法将索引 {idx} 处的状态数据转换为 NumPy，跳过此样本。错误: {e}")
                    continue # 跳过这个无法处理的样本

        # 检查是否成功收集到任何样本
        if not states_np:
            ERROR(logger, "未能从缓冲区收集有效的状态样本进行可视化。")
            return
            
        # 将 NumPy 状态列表转换为单个 NumPy 数组
        try:
            final_states_np = np.array(states_np)
        except ValueError as e:
            # 如果状态形状不一致，np.array 会失败
            ERROR(logger, f"收集的状态形状不一致，无法创建批处理 NumPy 数组。错误: {e}")
            return

        # 转换为张量并移到设备
        states_tensor = torch.tensor(final_states_np, dtype=torch.float32).to(self.device)

        # 计算Q值
        self.policy_net.eval()
        with torch.no_grad():
            # 这一步已经正确地将 Q 值移回 CPU 并转换为 NumPy
            q_values = self.policy_net(states_tensor).cpu().numpy()
        
        # 打印Q值分布
        INFO("\nQ值分布统计:")
        INFO(logger, f"平均Q值: {np.mean(q_values)}")
        INFO(logger, f"最大Q值: {np.max(q_values)}")
        INFO(logger, f"最小Q值: {np.min(q_values)}")
        INFO(logger, f"Q值范围: {np.max(q_values) - np.min(q_values)}")
        INFO(logger, f"Q值标准差: {np.std(q_values)}")
        
        # 如果Q值都很接近，可能表示网络没有学习
        if np.std(q_values) < 0.1:
            WARNING(logger, "警告: Q值标准差很小，可能表示网络没有有效学习")
        
        # 如果matplotlib可用，绘制Q值分布
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(q_values.flatten(), bins=50)
            plt.title('Q值分布')
            plt.savefig('q_values_distribution.png')
            INFO(logger, "Q值分布图已保存到 q_values_distribution.png")
        except ImportError:
            WARNING(logger, "未安装matplotlib，无法绘制分布图")

    def save(self, directory, filename):
        """保存模型和训练状态"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'use_prioritized_replay': self.use_prioritized_replay
        }, filepath)
        INFO(logger, f"模型已保存到 {filepath}")
        
    def load(self, filepath):
        """加载模型和训练状态"""
        if not os.path.exists(filepath):
            INFO(logger, f"模型文件不存在: {filepath}")
            return False
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        
        # 确保优先经验回放设置与加载的模型匹配
        saved_per = checkpoint.get('use_prioritized_replay', False)
        if saved_per != self.use_prioritized_replay:
            INFO(logger, f"警告: 加载的模型使用了{'优先'if saved_per else '普通'}经验回放，但当前配置使用{'优先'if self.use_prioritized_replay else '普通'}经验回放")
        
        INFO(logger, f"模型已加载: {filepath}, 训练步数: {self.steps_done}")
        return True
        
    def setEvalMode(self):
        """设置为评估模式"""
        self.policy_net.eval()
        
    def setTrainMode(self):
        """设置为训练模式"""
        self.policy_net.train()
        
    def getCurrentEpsilon(self):
        """获取当前的 epsilon 值"""
        return self._calculateEpsilon()
        
    def updateStepsDone(self, steps):
        """更新已完成的步数"""
        self.steps_done = steps 