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
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, _ = self.env.reset(**kwargs)
        return obs, {} # 返回 observation 和空的 info 字典

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """对于需要按 FIRE 键开始的游戏 (如 Breakout)，在 Reset 后执行 FIRE 操作"""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1) # FIRE
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2) # 第二个动作，通常是有效的开始动作
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}

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
            obs, _, _, _, info = self.env.step(0)
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
        return obs.astype(np.float32) / 255.0


def makeAtari(
        env_id, 
        max_episode_steps=None, 
        noop_max=30, 
        frame_skip=4, 
        screen_size=84, 
        terminal_on_life_loss=True, 
        clip_rewards=True, 
        frame_stack=4, 
        render_mode=None):
    """创建并包装 Atari 环境的辅助函数"""
    env = gym.make(env_id, obs_type='rgb', render_mode=render_mode) # 添加 render_mode 参数

    # 设置最大步数限制
    if max_episode_steps is not None:
         env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    assert 'NoFrameskip' in env.spec.id # 确保使用 NoFrameskip 版本
    env = NoopResetEnv(env, noop_max=noop_max)
    # MaxAndSkip 不再直接支持 TimeLimit, TimeLimit 应在最外层或不使用 MaxAndSkip
    env = MaxAndSkipEnv(env, skip=frame_skip) # 注意：MaxAndSkipEnv 应该在 TimeLimit 之前

    if terminal_on_life_loss:
        env = EpisodicLifeEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env, width=screen_size, height=screen_size) # 默认灰度

    if clip_rewards:
        env = ClipRewardEnv(env)

    if frame_stack > 1:
        env = FrameStack(env, frame_stack) # FrameStack 需要 CHW 格式输入，并在内部处理

    env = NormalizeObservation(env)

    return env 