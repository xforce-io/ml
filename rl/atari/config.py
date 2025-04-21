import torch

# --- 通用配置 ---
class GeneralConfig:
    ENV_NAME = "BreakoutNoFrameskip-v4"  # 要运行的 Atari 游戏
    DEVICE = None # 自动检测 MPS
    TOTAL_TRAINING_STEPS = 1000000     # 总训练步数 (减少以便快速演示)
    LOG_INTERVAL = 1000                # 日志打印频率
    SAVE_INTERVAL = 50000              # 模型保存频率
    MODEL_SAVE_DIR = "./saved_models"    # 模型保存路径
    RANDOM_SEED = 42                   # 随机种子

    # 评估参数
    EVAL_EPISODES = 100                # 评估时的 episode 数量
    EVAL_EPSILON = 0.01               # 评估时的固定探索率 (非常小)
    VICTORY_THRESHOLD = 10.0           # 计算胜率的奖励阈值

    def __init__(self):
        if torch.cuda.is_available():
            self.DEVICE = "cuda"
        elif torch.backends.mps.is_available():
            self.DEVICE = "mps"
        else:
            self.DEVICE = "cpu"

# --- DQN 算法配置 ---
class DqnConfig:
    REPLAY_BUFFER_CAPACITY = 100000
    LEARNING_RATE = 5e-5        # 增加学习率，加速学习
    GAMMA = 0.99
    BATCH_SIZE = 64               # 增大批次大小，使梯度更稳定
    TARGET_UPDATE_FREQUENCY = 5000  # 降低更新频率，更快地同步目标网络和策略网络
    EPSILON_START = 1.0
    EPSILON_END = 0.1             # 降低最终探索率，使算法更倾向于利用学到的策略
    EPSILON_DECAY_STEPS = 300000   # 增加探索率衰减的步数，使探索持续更长时间
    LEARNING_STARTS = 5000         # 开始学习的步数，减少以便快速启动学习
    GRAD_CLIP_VALUE = 5.0          # 梯度裁剪值
    
    # 优先经验回放参数
    USE_PRIORITIZED_REPLAY = True   # 是否使用优先经验回放
    PER_ALPHA = 0.6                 # 优先级的幂，控制优先级使用程度 (0 = 均匀采样, 1 = 完全按优先级采样)
    PER_BETA_START = 0.4            # 重要性采样的初始 beta 值 (0 = 无修正, 1 = 完全修正)
    PER_BETA_INCREMENT = 0.001      # 每次采样后 beta 的增量, 使得训练过程中 beta 逐渐增加到 1
    PER_EPSILON = 0.01              # 添加到 TD 误差的小常数, 确保每个经验都有被采样的机会

# --- 随机算法配置 (可能为空，或包含特定参数) ---
class RandomConfig:
    pass # 随机算法通常不需要额外参数

# --- 将所有配置组合到一个地方 (可选) ---
class Config:
    general = GeneralConfig()
    dqn = DqnConfig()
    random = RandomConfig()