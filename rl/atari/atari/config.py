import torch
import yaml
import os

# Define the absolute path to the config file
# __file__ is the path to the current script (config.py)
# os.path.dirname(__file__) gets the directory containing config.py (atari/)
# os.path.abspath resolves the absolute path
# os.path.join combines the directory path and the filename
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))

def load_config_from_file(config_name='global'):
    """从指定的配置文件加载配置"""
    config_path = os.path.join(CONFIG_DIR, f'{config_name}.yaml')
    config = {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:  # Handle empty file case
                config = {}
    except FileNotFoundError:
        # Handle the case where the config file doesn't exist
        print(f"Error: Configuration file not found at {config_path}")
        # You might want to raise an exception or provide default values here
        # raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        # Handle errors during YAML parsing
        print(f"Error parsing YAML file {config_path}: {e}")
        # Depending on your application's needs, you might want to raise the exception
        # raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        # Catch any other potential exceptions during file reading or processing
        print(f"An unexpected error occurred while loading config: {e}")
        # raise
    
    return config

# --- 通用配置 ---
class GeneralConfig:
    DEFAULT_VALUES = {
        "ENV_NAME": "BreakoutNoFrameskip-v4",
        "TOTAL_TRAINING_STEPS": 1000000,
        "LOG_INTERVAL": 1000,
        "SAVE_INTERVAL": 50000,
        "VIDEO_SAVE_INTERVAL": 20000,
        "MODEL_SAVE_DIR": "./saved_models",
        "VIDEO_SAVE_DIR": "./videos",
        "RANDOM_SEED": 42,
        "EVAL_EPISODES": 100,
        "EVAL_EPSILON": 0.01,
        "VICTORY_THRESHOLD": 10.0,
    }

    def __init__(self, loaded_config: dict):
        general_cfg = loaded_config.get('general', {})
        for key, default in self.DEFAULT_VALUES.items():
            if key not in general_cfg:
                setattr(self, key, default)
            else:
                setattr(self, key, general_cfg.get(key, default))

        # 动态设备检测
        if torch.cuda.is_available():
            self.DEVICE = "cuda"
        elif torch.backends.mps.is_available():
            self.DEVICE = "mps"
        else:
            self.DEVICE = "cpu"
        self.DEVICE = general_cfg.get('device', self.DEVICE)

# --- DQN 算法配置 ---
class DqnConfig:
    DEFAULT_VALUES = {
        "REPLAY_BUFFER_CAPACITY": 200000,
        "LEARNING_RATE": 1e-4,
        "GAMMA": 0.99,
        "BATCH_SIZE": 64,
        "TARGET_UPDATE_FREQUENCY": 5000,
        "EPSILON_START": 1.0,
        "EPSILON_END": 0.01,
        "EPSILON_DECAY_STEPS": 1000000,
        "LEARNING_STARTS": 50000,
        "GRAD_CLIP_VALUE": 5.0,
        "USE_PRIORITIZED_REPLAY": True,
        "PER_ALPHA": 0.6,
        "PER_BETA_START": 0.4,
        "PER_BETA_INCREMENT": 0.001,
        "PER_EPSILON": 0.01,
    }

    def __init__(self, loaded_config: dict):
        dqn_cfg = loaded_config.get('dqn', {})
        for key, default in self.DEFAULT_VALUES.items():
            if key not in dqn_cfg:
                setattr(self, key, default)
            else:
                setattr(self, key, dqn_cfg.get(key, default))

# --- 随机算法配置 (可能为空，或包含特定参数) ---
class RandomConfig:
    def __init__(self, loaded_config: dict):
        self.random_cfg = loaded_config.get('random', {})

# --- 将所有配置组合到一个地方 ---
class Config:
    def __init__(self, config_name='global'):
        loaded_config = load_config_from_file(config_name)
        self.general = GeneralConfig(loaded_config)
        self.dqn = DqnConfig(loaded_config)
        self.random = RandomConfig(loaded_config)

if __name__ == '__main__':
    config = Config()
    print("从 YAML 加载的配置（包含默认值）：")
    print("\n--- 通用配置 ---")
    for key, value in vars(config.general).items():
        if key != 'DEFAULT_VALUES':
            print(f"{key}: {value}")
    print("\n--- DQN 配置 ---")
    for key, value in vars(config.dqn).items():
        if key != 'DEFAULT_VALUES':
            print(f"{key}: {value}")
    print("\n--- 随机配置 ---")
    print("(无特定属性)")