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
    LEARNING_RATE = 2e-4        # 增加学习率，加速学习
    GAMMA = 0.99
    BATCH_SIZE = 256               # 增大批次大小，使梯度更稳定
    TARGET_UPDATE_FREQUENCY = 4000  # 降低更新频率，更快地同步目标网络和策略网络
    EPSILON_START = 1.0
    EPSILON_END = 0.2             # 降低最终探索率，使算法更倾向于利用学到的策略
    EPSILON_DECAY_STEPS = 300000   # 增加探索率衰减的步数，使探索持续更长时间
    LEARNING_STARTS = 5000         # 开始学习的步数，减少以便快速启动学习
    GRAD_CLIP_VALUE = 10.0          # 梯度裁剪值

# --- 随机算法配置 (可能为空，或包含特定参数) ---
class RandomConfig:
    pass # 随机算法通常不需要额外参数

# --- 将所有配置组合到一个地方 (可选) ---
class Config:
    general = GeneralConfig()
    dqn = DqnConfig()
    random = RandomConfig()

# # 示例如何使用:
# cfg = Config()
# print(f"设备: {cfg.general.DEVICE}")
# print(f"DQN 学习率: {cfg.dqn.LEARNING_RATE}") 