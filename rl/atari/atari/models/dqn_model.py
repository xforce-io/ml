import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """简单的卷积 Q 网络"""
    def __init__(self, input_shape, num_actions, game_type="atari"):
        """
        Args:
            input_shape (tuple): 输入观察值的形状
            num_actions (int): 可选动作的数量
            game_type (str): 游戏类型，可以是"atari"或"mario"
        """
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.game_type = game_type

        # 根据游戏类型设置输入形状
        if game_type == "mario":
            # Mario游戏使用4帧堆叠，确保通道数是4
            self.input_channels = 4
            self.input_height = 84
            self.input_width = 84
        else:
            # Atari游戏使用正常检测
            if len(input_shape) == 3:
                if input_shape[0] <= 4:  # 如果第一维是通道数（一般不会超过4）
                    self.input_channels = input_shape[0]
                    self.input_height = input_shape[1]
                    self.input_width = input_shape[2]
                else:  # 如果形状是(H, W, C)格式
                    self.input_channels = input_shape[2]
                    self.input_height = input_shape[0]
                    self.input_width = input_shape[1]
            else:
                raise ValueError(f"输入形状不正确: {input_shape}，需要是三维形状")

        # 根据游戏类型选择不同的卷积架构
        if game_type == "mario":
            # Mario使用较小的卷积核和步长
            # 对于84x84的输入，我们需要适当调整卷积参数
            self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        else:
            # Atari使用原始的Nature DQN架构
            self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 计算卷积层输出的大小，以便连接全连接层
        # 创建一个假的输入张量来计算尺寸
        dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        conv_out_size = self._get_conv_out(dummy_input)

        # 全连接层
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, x):
        """计算卷积层的输出尺寸"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x[0].numel()  # 排除批次维度

    def forward(self, x):
        """前向传播"""
        # 首先将像素值归一化到 [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # 通过卷积层和 ReLU 激活函数
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 通过全连接层和 ReLU 激活函数 (除了最后一层)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values 