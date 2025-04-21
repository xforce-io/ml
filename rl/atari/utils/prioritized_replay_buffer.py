from log import WARNING
import torch
import numpy as np
import random

class SumTree:
    """
    用于优先经验回放的和树数据结构。
    叶节点存储每个经验的优先级，内部节点存储子节点优先级之和。
    允许O(log n)时间复杂度的优先级采样。
    """
    def __init__(self, capacity, device):
        """
        初始化SumTree
        
        Args:
            capacity: 叶子节点的数量，即经验容量
            device: 存储设备
        """
        self.capacity = capacity  # 叶子节点的数量 (经验容量)
        self.device = torch.device(device)
        
        # 总节点数为 2 * capacity - 1 (完全二叉树特性)
        # 前 capacity-1 个是内部节点，后 capacity 个是叶子节点
        self.tree = torch.zeros(2 * capacity - 1, dtype=torch.float32, device=self.device)
        
        # 经验数据的写入位置（循环覆盖的位置）
        self.data_pointer = 0
        
        # 预计算树的层级数，用于向量化查找
        self._num_levels = int(np.ceil(np.log2(self.capacity)))

    def update(self, idx, priority):
        """
        更新叶子节点的优先级并传播变化到根节点
        
        Args:
            idx: 叶子节点的索引（范围为 0 到 capacity-1）
            priority: 新的优先级值
        """
        # 转换为树中的实际索引
        tree_idx = idx + self.capacity - 1
        
        # 计算变化值
        change = priority - self.tree[tree_idx]
        
        # 更新叶子节点
        self.tree[tree_idx] = priority
        
        # 向上传播变化到所有父节点
        while tree_idx != 0:  # 当还没到达根节点
            tree_idx = (tree_idx - 1) // 2  # 父节点索引
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        根据累积优先级获取叶子节点索引、优先级值和实际数据索引
        
        Args:
            v: 累积优先级值，范围为 [0, total_priority)
            
        Returns:
            (tree_idx, priority, data_idx)
        """
        parent_idx = 0
        
        # 遍历树直到叶子节点
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # 如果到达叶子节点，停止
            if left_child_idx >= len(self.tree):
                tree_idx = parent_idx
                break
                
            # 向下遍历树，根据累积优先级选择右边或左边的子节点
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        # 计算实际数据索引
        data_idx = tree_idx - (self.capacity - 1)
        
        return tree_idx, self.tree[tree_idx], data_idx

    def get_leaf_batch(self, v_batch):
        """
        根据一批累积优先级值，向量化地获取叶子节点索引和优先级值。

        Args:
            v_batch: 包含多个累积优先级值的张量，形状为 (batch_size,)

        Returns:
            (data_indices, priorities): 包含数据索引和对应优先级的张量
        """
        batch_size = v_batch.shape[0]
        # 初始化所有批次样本的当前节点索引为根节点 (0)
        indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        current_v = v_batch.clone() # 克隆以避免修改输入

        # --- 修改开始 ---
        # 跟踪哪些样本尚未到达叶子节点
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        # 预计算叶子节点的起始索引
        leaf_start_index = self.capacity - 1
        # --- 修改结束 ---

        # 迭代树的层级次，从根节点向下查找
        for _ in range(self._num_levels):
            # --- 修改开始 ---
            # 如果所有样本都已到达叶子节点，则提前退出
            if not active_mask.any():
                break

            # 只对活跃的样本进行操作
            current_indices = indices[active_mask]
            current_v_active = current_v[active_mask]

            # 计算活跃样本的左子节点索引
            # 由于 current_indices 保证是内部节点 (< leaf_start_index)，
            # 因此 left_child_indices 会小于 2*capacity - 1，访问安全
            left_child_indices = 2 * current_indices + 1
            left_child_priorities = self.tree[left_child_indices]

            # 判断活跃样本是向左还是向右
            go_left_mask_active = current_v_active <= left_child_priorities
            go_right_mask_active = ~go_left_mask_active

            # 准备活跃样本的更新后索引
            updated_indices_active = torch.zeros_like(current_indices)
            updated_indices_active[go_left_mask_active] = left_child_indices[go_left_mask_active]
            # 右子节点索引
            updated_indices_active[go_right_mask_active] = left_child_indices[go_right_mask_active] + 1

            # 更新向右移动的样本的 v 值
            current_v_active[go_right_mask_active] -= left_child_priorities[go_right_mask_active]

            # 将更新后的值放回原张量
            indices[active_mask] = updated_indices_active
            current_v[active_mask] = current_v_active

            # 更新 active_mask：如果样本的新索引已经是叶子节点，则将其标记为非活跃
            # 注意：这里需要用 indices[active_mask] 获取更新后的索引来判断
            active_mask_indices = torch.where(active_mask)[0] # 获取当前 active_mask 为 True 的原始索引
            active_mask[active_mask_indices] = indices[active_mask_indices] < leaf_start_index
            # --- 修改结束 ---

        # 循环结束后，indices 张量包含了所有样本对应的叶子节点在树中的索引
        tree_indices = indices
        # 计算对应的数据索引 (在原始存储中的索引)
        data_indices = tree_indices - leaf_start_index # 使用预计算的 leaf_start_index
        # 获取最终的叶子节点优先级 (此时 tree_indices 均为叶子节点索引，访问安全)
        priorities = self.tree[tree_indices]

        return data_indices, priorities

    def total_priority(self):
        """
        返回根节点的值，即所有优先级之和
        """
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区，使用SumTree实现
    支持按TD误差的优先级进行采样
    """
    def __init__(self, capacity, device='mps', alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """
        初始化优先经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            device: 存储设备 ('cpu', 'cuda', 或 'mps')
            alpha: 优先级使用程度 [0, 1] (0表示均匀采样，1表示完全优先级采样)
            beta: 重要性采样权重指数 [0, 1] (0表示无修正，1表示完全修正)
            beta_increment: 每次采样后beta的增量
            epsilon: 加到TD误差上的小常数，确保所有经验都有被采样的机会
        """
        self.capacity = capacity
        self.device = torch.device(device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # 用于新经验的初始最大优先级
        
        # 初始化SumTree
        self.sum_tree = SumTree(capacity, device)
        
        # 缓冲区变量初始化
        self.current_size = 0
        self.ptr = 0  # 指向下一个要插入的位置
        
        # 缓冲区中的张量将在_initialize_buffers中分配
        self.states = None
        self.next_states = None
        
        # 在目标设备上预分配固定大小的张量
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
    
    def _initialize_buffers(self, state):
        """根据第一个状态在目标设备上初始化缓冲区"""
        # 确保state是NumPy数组以获取形状
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
        elif not isinstance(state, np.ndarray):
            state_np = np.array(state)
        else:
            state_np = state
            
        state_shape = state_np.shape
        # 在目标设备上分配状态缓冲区
        self.states = torch.zeros((self.capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.capacity, *state_shape), dtype=torch.float32, device=self.device)
        INFO(f"PrioritizedReplayBuffer在设备{self.device}上初始化完成，状态形状: {state_shape}")

    def push(self, state, action, reward, next_state, done, td_error=None):
        """
        存储一个经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否为终止状态
            td_error: TD误差，如果为None则使用最大优先级
        """
        # 如果是第一次添加数据，初始化缓冲区
        if self.states is None:
            self._initialize_buffers(state)
        
        # 将输入数据转换为张量并移动到目标设备
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action_tensor = torch.as_tensor(action, dtype=torch.long).to(self.device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32).to(self.device)
        
        # 计算优先级并添加到sum_tree
        if td_error is None:
            priority = self.max_priority  # 新经验使用最大优先级
        else:
            # 从TD误差计算优先级
            td_error = abs(td_error) if isinstance(td_error, (int, float)) else abs(td_error.item())
            priority = (td_error + self.epsilon) ** self.alpha
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
        
        # 存储到目标设备上的缓冲区
        self.states[self.ptr] = state_tensor
        self.actions[self.ptr] = action_tensor
        self.rewards[self.ptr] = reward_tensor
        self.next_states[self.ptr] = next_state_tensor
        self.dones[self.ptr] = done_tensor
        
        # 更新sum_tree
        self.sum_tree.update(self.ptr, priority)
        
        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        从缓冲区中按优先级采样一个批次的经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, actions, rewards, next_states, dones, indices, is_weights)
                - indices: 采样的经验索引，用于后续更新优先级
                - is_weights: 重要性采样权重，用于修正梯度偏差
        """
        if self.current_size < batch_size:
            raise ValueError(f"缓冲区经验数量不足 ({self.current_size} < {batch_size})")
        
        batch_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        batch_priorities = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        is_weights = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        # 计算优先级区间大小
        total_p = self.sum_tree.total_priority()
        # 检查总优先级是否有效
        if not torch.isfinite(total_p) or total_p <= 1e-6: # 增加一个小的阈值防止接近0
            WARNING(f"SumTree总优先级异常: {total_p}. 无法进行优先采样，可能需要检查优先级更新逻辑。")
            # 在这种情况下，可能需要采取备用策略，例如均匀采样或抛出错误
            # 这里我们暂时抛出错误，因为优先采样已失效
            raise ValueError(f"SumTree总优先级异常 ({total_p})，无法采样")
            
        # 增加beta以逐渐接近1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # --- 优化点：向量化采样 ---
        # 1. 计算每个区间的边界
        segment_boundaries = torch.linspace(0, total_p, batch_size + 1, device=self.device)
        lower_bounds = segment_boundaries[:-1]
        upper_bounds = segment_boundaries[1:]
        
        # 2. 在每个区间内随机生成 v 值 (一次性生成 batch_size 个)
        v_batch = lower_bounds + torch.rand(batch_size, device=self.device) * (upper_bounds - lower_bounds)

        # 3. 使用向量化的 get_leaf_batch 获取索引和优先级
        batch_indices, batch_priorities = self.sum_tree.get_leaf_batch(v_batch)
        # --- 优化结束 ---

        # 计算重要性采样权重
        # P(j) = p_j^α / sum_i p_i^α
        # 权重 w_j = (1/N * 1/P(j))^β
        sampling_probabilities = batch_priorities / total_p
        # 限制概率的最小值，防止后续计算除以零或负数开方
        sampling_probabilities = torch.clamp(sampling_probabilities, min=1e-8)
        
        is_weights = (self.current_size * sampling_probabilities) ** (-self.beta)
        
        # 归一化权重到[0, 1]范围，并检查最大权重
        max_weight = torch.max(is_weights)
        if not torch.isfinite(max_weight) or max_weight <= 1e-8:
            WARNING(f"重要性采样权重最大值异常: {max_weight}. 将权重设置为1。")
            # 如果最大权重无效，将所有权重设为1（相当于均匀采样）
            is_weights = torch.ones_like(is_weights)
        else:
            is_weights /= max_weight
            
        # (可选) 对最终权重进行限制，防止极端值影响损失
        is_weights = torch.clamp(is_weights, min=1e-6, max=100.0) 
        
        # 根据索引获取数据
        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        next_states = self.next_states[batch_indices]
        dones = self.dones[batch_indices]
        
        return states, actions, rewards, next_states, dones, batch_indices, is_weights.unsqueeze(1)
    
    def update_priorities(self, indices, td_errors):
        """
        基于新的TD误差更新经验的优先级
        
        Args:
            indices: 要更新的经验索引
            td_errors: 对应的TD误差
        """
        for idx, td_error in zip(indices, td_errors):
            # 转为标量以便计算
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            error = abs(td_error.item()) if torch.is_tensor(td_error) else abs(td_error)
            
            # 检查 TD 误差是否有限
            if not np.isfinite(error):
                WARNING(f"TD误差包含非有限值: {error} at index {idx}. 使用epsilon作为优先级.")
                # 如果TD误差无效，使用一个基于epsilon的默认优先级
                priority = self.epsilon ** self.alpha
            else:
                # 计算新的优先级
                 priority = (error + self.epsilon) ** self.alpha

            # 确保优先级是有限的正数
            if not np.isfinite(priority):
                WARNING(f"计算出的优先级为非有限值 at index {idx}. 使用epsilon作为优先级.")
                priority = self.epsilon ** self.alpha # 再次确保
            # 限制优先级的最小值，防止为0或负数
            priority = torch.clamp(torch.tensor(priority), min=1e-6).item() # Clamp后转回float存储
            
            # 更新sum_tree
            self.sum_tree.update(idx, priority)
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """返回当前缓冲区中的经验数量"""
        return self.current_size 