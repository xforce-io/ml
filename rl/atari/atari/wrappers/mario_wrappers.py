import numpy as np
import cv2
from collections import deque
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import random
import gymnasium

cv2.ocl.setUseOpenCL(False)

def makeMario(
        worldStage=None,
        maxEpisodeSteps=10000,
        frameSkip=4,
        grayscale=True,
        resizeShape=(84, 84),
        frameStack=4,
        normalizeObs=True,
        renderMode=None
):
    """创建Super Mario Bros环境的辅助函数
    
    Args:
        worldStage: 游戏关卡, 例如"1-1"表示第一世界第一关
        maxEpisodeSteps: 每个episode的最大步数
        frameSkip: 要跳过的帧数
        grayscale: 是否将图像转换为灰度
        resizeShape: 调整图像大小的形状
        frameStack: 要堆叠的帧数
        normalizeObs: 是否归一化观察值
        renderMode: 渲染模式
    """
    # 确定环境ID
    if worldStage:
        env_id = f"SuperMarioBros-{worldStage}-v0"
    else:
        env_id = "SuperMarioBros-v0"
    
    # 使用官方方法创建环境
    print(f"创建 Mario 环境: {env_id}")
    
    # 直接导入原始环境创建函数，绕过gym的包装器
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
    from nes_py.wrappers import JoypadSpace
    
    # 检查是否指定了关卡
    if worldStage:
        world, stage = worldStage.split('-')
        base_env = SuperMarioBrosEnv(
            rom_mode='vanilla',
            lost_levels=False,
            target=(int(world), int(stage))
        )
    else:
        base_env = SuperMarioBrosEnv(
            rom_mode='vanilla',
            lost_levels=False
        )
    
    # 应用JoypadSpace包装器，简化动作空间
    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    
    # 创建我们自己的适配器，自己处理时间限制
    mario_env = MarioEnv(
        env=env,
        maxEpisodeSteps=maxEpisodeSteps,
        frameSkip=frameSkip,
        grayscale=grayscale,
        resizeShape=resizeShape,
        frameStack=frameStack,
        normalizeObs=normalizeObs,
        renderMode=renderMode
    )
    
    return mario_env

class MarioEnv(gymnasium.Env):
    """包装器，将旧版gym接口适配为gymnasium风格的接口"""
    def __init__(self, env, maxEpisodeSteps=10000, frameSkip=4, 
                 grayscale=True, resizeShape=(84, 84), frameStack=4, 
                 normalizeObs=True, renderMode=None):
        """初始化环境"""
        super().__init__()
        
        # 保存最外层的环境引用
        self.wrapped_env = env 
        
        # 分析环境层次结构
        print("初始化MarioEnv适配器，环境层次结构:")
        self._debug_env_hierarchy(env)
        
        # 从构造函数接收参数
        self.max_steps = maxEpisodeSteps
        self.frameSkip = frameSkip
        self.grayscale = grayscale
        self.resizeShape = resizeShape
        self.normalizeObs = normalizeObs
        
        # 设置动作空间
        self.action_space = env.action_space
        
        # 设置观察空间 - 使用gymnasium的Box
        obs_shape = resizeShape[::-1] if grayscale else (*resizeShape, 3)  # (H, W) 如果是灰度，(H, W, 3) 如果是彩色
        low = 0.0 if normalizeObs else 0
        high = 1.0 if normalizeObs else 255
        dtype = np.float32 if normalizeObs else np.uint8
        
        # 如果使用帧堆栈，调整观察空间形状
        if frameStack > 1:
            if grayscale:  # 灰度图像的堆叠，形状为(stack_size, height, width)
                obs_shape = (frameStack, *obs_shape)
            else:  # 彩色图像的堆叠，形状为(stack_size, height, width, 3)
                obs_shape = (frameStack, *obs_shape)
        
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, shape=obs_shape, dtype=dtype
        )
        
        # 添加metadata属性
        self.metadata = getattr(env, 'metadata', {'render_modes': ['human', 'rgb_array'], 'render_fps': 30})
        
        # 添加render_mode属性
        self.render_mode = renderMode if renderMode else "rgb_array"
        self.render_fps = self.metadata.get('render_fps', 30)
        
        # 设置帧堆栈
        self.stack_size = frameStack
        self.frames = deque([], maxlen=frameStack)
        
        # 是否需要渲染
        self.should_render = renderMode == "human"
        
        # 当前步数，用于实现时间限制
        self.current_steps = 0
        
    def _get_base_env(self, env):
        """递归查找底层环境，绕过所有包装器"""
        if hasattr(env, 'env'):
            return self._get_base_env(env.env)
        return env
    
    def _debug_env_hierarchy(self, env, depth=0):
        """打印环境包装器层级，用于调试"""
        print(f"{'  ' * depth}环境层级 {depth}: {type(env).__name__}")
        if hasattr(env, 'env'):
            self._debug_env_hierarchy(env.env, depth + 1)
    
    def preprocess_obs(self, obs):
        """预处理观察值"""
        # 转换为灰度图像
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # 调整大小
        if self.resizeShape:
            obs = cv2.resize(obs, self.resizeShape, interpolation=cv2.INTER_AREA)
        
        # 归一化
        if self.normalizeObs:
            obs = obs.astype(np.float32) / 255.0
        
        return obs
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        # 处理seed参数
        if seed is not None and hasattr(self.wrapped_env, 'seed'):
            self.wrapped_env.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        # 重置环境 - 兼容旧版gym接口
        obs = self.wrapped_env.reset()
        
        # 重置计数器
        self.current_steps = 0
        
        # 预处理观察值
        processed_obs = self.preprocess_obs(obs)
        
        # 重置帧堆栈
        for _ in range(self.stack_size):
            self.frames.append(processed_obs)
        
        # 渲染（如果需要）
        if self.should_render:
            self.wrapped_env.render()
        
        # 返回堆叠后的观察值和空的info字典（适配gymnasium接口）
        return self.get_stacked_obs(), {}
        
    def step(self, action):
        """执行动作 - 从旧版gym API (4个返回值) 转换到新版gymnasium API (5个返回值)"""
        total_reward = 0
        done = False
        info = {}
        
        # 跟踪当前步数
        self.current_steps += 1
        
        # 检查是否超过最大步数
        timeout = self.current_steps >= self.max_steps
        
        # 帧跳过 - 执行原始环境的step
        for _ in range(self.frameSkip):
            # gym-super-mario-bros使用旧版gym API，返回4个值
            obs, reward, game_done, info = self.wrapped_env.step(action)
            total_reward += reward
            
            # 渲染（如果需要）
            if self.should_render:
                self.wrapped_env.render()
            
            # 如果游戏结束，提前终止帧跳过
            if game_done:
                done = True
                break
        
        # 预处理观察值
        processed_obs = self.preprocess_obs(obs)
        
        # 更新帧堆栈
        self.frames.append(processed_obs)
        
        # 适配gymnasium API，将done拆分为terminated和truncated
        # 如果游戏自然结束，而非超时导致的结束
        terminated = done and not timeout
        # 如果是超时导致的结束
        truncated = timeout
        
        # 返回堆叠后的观察值、总奖励、终止标志、截断标志和info字典（完全符合gymnasium接口）
        return self.get_stacked_obs(), total_reward, terminated, truncated, info
        
    def get_stacked_obs(self):
        """获取堆叠后的观察值"""
        # 对于帧堆栈，将所有帧沿着新的轴堆叠
        stacked_frames = np.stack(list(self.frames), axis=0)
        return stacked_frames
        
    def close(self):
        """关闭环境"""
        try:
            return self.wrapped_env.close()
        except Exception as e:
            # 忽略"env has already been closed"错误
            print(f"Warning: {e}")
            return None
        
    def render(self):
        """渲染环境"""
        # This method is called by RecordVideo or for human display based on self.render_mode
        if self.render_mode == "human":
            return self.wrapped_env.render(mode="human")
        elif self.render_mode == "rgb_array":
            # This is called by RecordVideo.capture_frame() -> env.render()
            # Underlying env.render("rgb_array") returns a direct reference to internal screen buffer.
            # A copy must be returned to prevent all frames in the video recorder's list
            # from pointing to the same mutable buffer, which would result in a static video
            # of the last frame.
            current_screen = self.wrapped_env.render(mode="rgb_array")
            if current_screen is None:
                return None 
            return np.copy(current_screen)
        else:
            # Fallback for any other unhandled render_mode, attempting rgb_array with a copy.
            # This case should ideally not be reached if configured correctly for video recording.
            current_screen = self.wrapped_env.render(mode="rgb_array")
            return np.copy(current_screen) if current_screen is not None else None 