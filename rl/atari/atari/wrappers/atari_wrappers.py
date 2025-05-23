import gymnasium as gym
import numpy as np
import cv2
from collections import deque

cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """在 Reset 时执行随机次数的 No-op 操作"""
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            # 新版gymnasium API返回5个值
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """对于需要按 FIRE 键开始的游戏 (如 Breakout)，在 Reset 后执行 FIRE 操作"""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
            
        # 新版gymnasium API
        obs, _, terminated, truncated, info = self.env.step(1) # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
                
        obs, _, terminated, truncated, info = self.env.step(2) # 第二个动作，通常是有效的开始动作
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
                
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """将损失一条生命视为 episode 结束信号"""
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # 检查生命数
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # 损失生命但游戏未结束，强制 terminated=True
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """仅在真正的 episode 结束时 Reset，否则只重置生命计数器"""
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # 不调用 env.reset()，只发送 NOOP
            obs, _, terminated, truncated, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """合并最后两帧并跳过指定数量的帧"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """将奖励裁剪到 +1, -1 或 0"""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """将图像帧转换为灰度图并调整大小"""
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        # 确保frame是numpy数组
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
            
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """将最近的 k 帧堆叠起来作为观察值"""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[2] * k, shp[0], shp[1])), dtype=env.observation_space.dtype) # 通道优先
        
        # 预分配数组以提高性能，避免频繁分配内存
        self.stacked_frames = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
            
        # 转换为 CHW (通道, 高, 宽) 格式
        obs_chw = np.transpose(obs, (2, 0, 1))
        for _ in range(self.k):
            self.frames.append(obs_chw)
        
        # 初始化 stacked_frames
        if self.stacked_frames is None:
            self.stacked_frames = np.concatenate(list(self.frames), axis=0)
        else:
            # 如果已存在，则更新内容
            idx = 0
            for frame in self.frames:
                frame_channels = frame.shape[0]
                self.stacked_frames[idx:idx+frame_channels] = frame
                idx += frame_channels
        
        return self.stacked_frames, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 转换为 CHW 格式
        obs_chw = np.transpose(obs, (2, 0, 1))
        self.frames.append(obs_chw)
        
        # 更新 stacked_frames
        idx = 0
        for frame in self.frames:
            frame_channels = frame.shape[0]
            self.stacked_frames[idx:idx+frame_channels] = frame
            idx += frame_channels
        
        return self.stacked_frames, reward, terminated, truncated, info


class NormalizeObservation(gym.ObservationWrapper):
    def observation(self, obs):
        # 确保输入是 numpy 数组
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        return obs.astype(np.float32) / 255.0


def makeAtari(
        envId,
        maxEpisodeSteps=None,
        noopMax=30,
        frameSkip=4,
        screenSize=84,
        terminalOnLifeLoss=True,
        clipRewards=True,
        frameStack=4,
        renderMode=None):
    """创建并包装 Atari 环境的辅助函数
    
    Args:
        envId (str): Atari环境ID
        maxEpisodeSteps (int, optional): 每个episode的最大步数
        noopMax (int): 执行的最大no-op操作数量
        frameSkip (int): 跳过的帧数
        screenSize (int): 图像的大小
        terminalOnLifeLoss (bool): 是否将生命损失视为episode结束
        clipRewards (bool): 是否将奖励裁剪至[-1,1]
        frameStack (int): 堆叠的帧数
        renderMode (str, optional): 渲染模式
    
    Returns:
        gym.Env: 包装后的Atari环境
    """
    # 转换环境名称 - 使用新版gymnasium中的ALE命名空间
    if 'NoFrameskip' in envId:
        # 如果包含NoFrameskip，则转换为ALE版本
        game = envId.split('NoFrameskip')[0]
        processedEnvId = f"ALE/{game}NoFrameskip-v5"
    elif 'ALE/' in envId:
        # 如果已经是ALE格式，保持原样
        processedEnvId = envId
    else:
        # 如果是简单游戏名称，使用ALE命名空间
        processedEnvId = f"ALE/{envId}-v5"
    
    try:
        # 尝试创建环境
        env = gym.make(processedEnvId, render_mode=renderMode)
        print(f"成功创建环境: {processedEnvId}")
    except Exception as e:
        print(f"创建环境 {processedEnvId} 失败: {e}")
        # 尝试不同的环境名称格式
        try:
            # 尝试老版本格式
            fallback_id = f"{envId.replace('NoFrameskip-v4', '-v4')}"
            print(f"尝试使用备选环境 ID: {fallback_id}")
            env = gym.make(fallback_id, render_mode=renderMode)
        except Exception as e2:
            print(f"创建备选环境 {fallback_id} 也失败: {e2}")
            # 最后尝试原始 ID
            print(f"使用原始环境 ID: {envId}")
            env = gym.make(envId, render_mode=renderMode)
    
    # 应用核心 wrappers
    env = NoopResetEnv(env, noop_max=noopMax)
    env = MaxAndSkipEnv(env, skip=frameSkip) # MaxAndSkipEnv 应在 TimeLimit 之前

    if terminalOnLifeLoss:
        env = EpisodicLifeEnv(env)

    # 检查 'FIRE' 动作是否存在且有效
    actionMeanings = env.unwrapped.get_action_meanings()
    if 'FIRE' in actionMeanings and actionMeanings.index('FIRE') == 1:
         # 确保 FIRE 是第二个动作，以匹配 FireResetEnv 的逻辑
        env = FireResetEnv(env)

    env = WarpFrame(env, width=screenSize, height=screenSize) # 默认灰度

    if clipRewards:
        env = ClipRewardEnv(env)

    if frameStack > 1:
        env = FrameStack(env, frameStack) # FrameStack 需要 CHW 格式输入，并在内部处理

    env = NormalizeObservation(env)

    # 在所有其他 wrappers 之后应用 TimeLimit
    if maxEpisodeSteps is not None:
         env = gym.wrappers.TimeLimit(env, max_episode_steps=maxEpisodeSteps)

    return env