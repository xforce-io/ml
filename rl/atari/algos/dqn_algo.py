import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import time

from .base_algo import Algo
from models.dqn_model import DQN             # 假设 dqn_model.py 在项目根目录
from utils.replay_buffer import ReplayBuffer # 假设 replay_buffer.py 在项目根目录

class AlgoDQN(Algo):
    """DQN 算法实现"""

    DEBUG_INTERVAL = 1000
    
    def __init__(self, env, config, device):
        super().__init__(env, config, device)
        print(f"初始化 DQN 算法 (AlgoDQN) 在设备: {self.device}")

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

        # 创建策略网络和目标网络
        self.policy_net = DQN(self.input_shape, self.num_actions).to(self.device)
        self.target_net = DQN(self.input_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=dqn_config.LEARNING_RATE)

        # 创建经验回放缓冲区
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
                print(f"[等待学习] 步数: {self.steps_done}/{self.learning_starts}, 经验缓冲区: {len(self.memory)}/{self.batch_size}")
            return None # 如果不满足条件，则暂时不学习，直接返回
            
        if len(self.memory) < self.batch_size:
            if self.steps_done % self.DEBUG_INTERVAL == 0:
                print(f"[缓冲区填充中] 经验缓冲区: {len(self.memory)}/{self.batch_size}")
            return None # 样本不足，不学习

        # --- 如果满足学习条件，则执行以下步骤 ---

        # 3. 从缓冲区采样
        t0 = time.time()
        batch = self.memory.sample(self.batch_size)
        if batch is None: # 以防万一采样失败
            print("[警告] 采样失败")
            return None
        # 解包批次数据
        states, actions, rewards, next_states, dones = batch
        buffer_time += time.time() - t0

        # 4. 计算 Q(s_t, a_t) - 当前状态-动作对的 Q 值 (预测值)
        t0 = time.time()
        self.policy_net.train() # 确保策略网络处于训练模式（启用 dropout, batchnorm 等）
        # 将采样到的 'states' 输入策略网络，得到这些状态下所有可能动作的 Q 值
        q_values = self.policy_net(states)
        # 从 q_values 中，根据采样到的 'actions'，精确地提取出实际执行动作对应的 Q 值
        q_values_for_actions = q_values.gather(1, actions)

        # 5. 计算 V(s_{t+1}) - 下一个状态的最大 Q 值 (用于构建目标值)
        with torch.no_grad(): # 不需要为目标网络计算梯度
            self.target_net.eval() # 确保目标网络处于评估模式
            # 将采样到的 'next_states' 输入目标网络
            next_q_values = self.target_net(next_states)
            # 对每个 next_state，找到其所有动作中 Q 值的最大值
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            # 计算目标 Q 值 (TD Target): R + γ * max_a' Q_target(s', a')
            # (1 - dones) 的作用是：如果一个状态是终止状态 (done=True)，那么其后续价值为 0
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        network_time = time.time() - t0

        # 6. 计算损失 (Loss)
        t0 = time.time()
        # 计算预测 Q 值 (q_values_for_actions) 和目标 Q 值 (target_q_values) 之间的差距
        # smooth_l1_loss (Huber Loss) 对异常值不那么敏感，比 MSELoss 更鲁棒
        loss = F.smooth_l1_loss(q_values_for_actions, target_q_values)

        # 7. 优化模型 (执行反向传播和参数更新)
        self.optimizer.zero_grad() # 清除上一轮的梯度
        loss.backward() # 计算损失相对于策略网络参数的梯度
        if self.grad_clip_value is not None: # 如果设置了梯度裁剪值
            # 对策略网络的梯度进行裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip_value)
        self.optimizer.step() # 使用优化器 (例如 Adam) 根据梯度更新策略网络的参数
        optimizer_time = time.time() - t0

        # 8. 定期更新目标网络
        # 检查当前总步数距离上次更新目标网络是否超过了指定频率 (target_update_frequency)
        if self.steps_done - self.last_target_update >= self.target_update_frequency:
            print(f"\n[Step {self.steps_done}] 更新目标网络")
            # 将策略网络的最新权重复制给目标网络
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # 更新上次更新的步数记录
            self.last_target_update = self.steps_done
            
        # 9. 记录损失值用于统计
        loss_value = loss.item()
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)
            
        # 10. 添加定期调试信息
        if self.steps_done % self.DEBUG_INTERVAL == 0:
            print(f"\n[DQN性能分析 Step {self.steps_done}]")
            print(f"  - 缓冲区操作时间: {buffer_time*1000:.1f}ms")
            print(f"  - 网络计算时间: {network_time*1000:.1f}ms")
            print(f"  - 优化器时间: {optimizer_time*1000:.1f}ms")
            print(f"  - 总时间: {(buffer_time + network_time + optimizer_time)*1000:.1f}ms")
            
            # 其他调试信息
            avg_reward = np.mean(rewards.cpu().numpy())
            avg_max_q = np.mean(q_values.max(dim=1)[0].detach().cpu().numpy())
            avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0
            print(f"  - 平均奖励: {avg_reward:.4f}")
            print(f"  - 平均最大Q值: {avg_max_q:.4f}")
            print(f"  - 平均损失: {avg_loss:.6f}")
            print(f"  - 当前Epsilon: {self._calculateEpsilon():.4f}")
            print(f"  - 缓冲区大小: {len(self.memory)}")
            print(f"  - 批次大小: {states.shape[0]}")
            
        return loss_value

    def visualizeQDistribution(self, num_samples=10):
        """可视化Q值分布来检查学习进展"""
        if len(self.memory) < num_samples:
            print("经验回放缓冲区样本不足")
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
                    print(f"警告：无法将索引 {idx} 处的状态数据转换为 NumPy，跳过此样本。错误: {e}")
                    continue # 跳过这个无法处理的样本

        # 检查是否成功收集到任何样本
        if not states_np:
            print("错误：未能从缓冲区收集有效的状态样本进行可视化。")
            return
            
        # 将 NumPy 状态列表转换为单个 NumPy 数组
        try:
            final_states_np = np.array(states_np)
        except ValueError as e:
            # 如果状态形状不一致，np.array 会失败
            print(f"错误：收集的状态形状不一致，无法创建批处理 NumPy 数组。错误: {e}")
            return

        # 转换为张量并移到设备
        states_tensor = torch.tensor(final_states_np, dtype=torch.float32).to(self.device)

        # 计算Q值
        self.policy_net.eval()
        with torch.no_grad():
            # 这一步已经正确地将 Q 值移回 CPU 并转换为 NumPy
            q_values = self.policy_net(states_tensor).cpu().numpy()
        
        # 打印Q值分布
        print("\nQ值分布统计:")
        print(f"平均Q值: {np.mean(q_values)}")
        print(f"最大Q值: {np.max(q_values)}")
        print(f"最小Q值: {np.min(q_values)}")
        print(f"Q值范围: {np.max(q_values) - np.min(q_values)}")
        print(f"Q值标准差: {np.std(q_values)}")
        
        # 如果Q值都很接近，可能表示网络没有学习
        if np.std(q_values) < 0.1:
            print("警告: Q值标准差很小，可能表示网络没有有效学习")
        
        # 如果matplotlib可用，绘制Q值分布
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(q_values.flatten(), bins=50)
            plt.title('Q值分布')
            plt.savefig('q_values_distribution.png')
            print("Q值分布图已保存到 q_values_distribution.png")
        except ImportError:
            print("未安装matplotlib，无法绘制分布图")

    def save(self, directory, filename):
        """保存策略网络的权重"""
        super().save(directory, filename) # 调用基类确保目录存在
        filepath = os.path.join(directory, filename)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"DQN 模型已保存到 {filepath}")
        return True

    def load(self, filepath):
        """加载策略网络的权重"""
        if not os.path.exists(filepath):
            print(f"错误：无法加载模型，文件不存在: {filepath}")
            return False
        try:
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict()) # 保持目标网络同步
            self.policy_net.eval() # 通常加载后用于评估，设置为评估模式
            self.target_net.eval()
            print(f"DQN 模型已从 {filepath} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False

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
        """更新内部步数计数器，可能触发目标网络更新"""
        self.steps_done = steps 