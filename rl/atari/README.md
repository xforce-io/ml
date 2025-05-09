# 强化学习环境

本项目支持在Atari和Super Mario Bros环境中训练和评估强化学习算法。

## 安装依赖

### 基本依赖

```bash
# 安装基本依赖
pip install -r requirements.txt
```

### Atari 环境安装（最新版gymnasium）

从gymnasium 1.0开始，Atari游戏环境被移到独立的包中，需要额外的安装步骤：

```bash
# 安装gymnasium及Atari扩展
pip install -r requirements-atari.txt

# 下载Atari ROM文件（必需步骤）
autorom --accept-license
```

如果使用的是zsh，请注意需要用引号包裹方括号，以避免zsh解释方括号作为通配符。

**注意**：如果遇到"Namespace ALE not found"错误，说明Atari Learning Environment未正确安装或ROM未下载。请确保执行了上述两个命令。

## API兼容性说明

本项目已更新以支持最新版本的gymnasium API（gymnasium v0.29+）。主要变化包括：

1. 环境名称从`BreakoutNoFrameskip-v4`变为`ALE/Breakout-v5`格式
2. 环境step方法现在返回5个值`(obs, reward, terminated, truncated, info)`而不是旧版的4个值
3. 环境reset方法现在始终返回`(obs, info)`元组

如果遇到`too many values to unpack`或`not enough values to unpack`错误，通常是因为API版本不兼容。

## 内存管理

如果遇到内存不足错误（例如`Invalid buffer size: XX GB`），可以通过修改`config/breakout.yaml`文件中的以下参数降低内存使用：

```yaml
dqn:
  REPLAY_BUFFER_CAPACITY: 20000  # 减小缓冲区大小（默认200000）
  BATCH_SIZE: 32                 # 减小批次大小（默认64）
```

## 使用方法

### 1. Atari游戏

训练DQN算法在Breakout游戏上：

```bash
python -m atari.main --game-type atari --env Breakout --algo dqn --mode train --steps 1000000
```

评估训练好的模型：

```bash
python -m atari.main --game-type atari --env Breakout --algo dqn --mode eval --render
```

录制游戏视频（每10000步保存一次）：

```bash
python -m atari.main --game-type atari --env Breakout --algo dqn --mode train --steps 1000000 --record-video
```

### 2. Super Mario Bros游戏

由于gym-super-mario-bros与当前环境中的gym和gymnasium库可能存在冲突，建议通过以下步骤创建一个专用虚拟环境来运行Mario：

```bash
# 创建一个新的venv环境 (例如命名为 .venv-mario)
python3 -m venv .venv-mario

# 激活环境
# macOS / Linux:
source .venv-mario/bin/activate
# Windows:
# .\.venv-mario\Scripts\activate

# (在新环境中) 使用 requirements-mario.txt 安装依赖
pip install -r requirements-mario.txt
```

确保您在**激活的 `.venv-mario` 环境**中运行Mario相关脚本：

```bash
# (在激活的 .venv-mario 环境中)
# 使用随机动作在1-1关卡运行一个episode
python -m atari.main --game-type mario --world-stage 1-1 --algo random --mode eval --render --episodes 1

# 指定世界和关卡
python -m atari.main --game-type mario --world-stage 1-2 --algo random --mode eval --render --episodes 1

# 在1-1关卡训练DQN算法
python -m atari.main --game-type mario --world-stage 1-1 --algo dqn --mode train --steps 1000000

# 训练并同时定期评估模型
python -m atari.main --game-type mario --world-stage 1-1 --algo dqn --mode train_and_eval --steps 1000000

# 使用训练好的DQN模型评估
python -m atari.main --game-type mario --world-stage 1-1 --algo dqn --mode eval --render --episodes 1

# 玩完后可以退出虚拟环境
deactivate
```

## 环境依赖版本

### Atari 环境

```
gymnasium>=0.29.1
ale-py==0.8.1
AutoROM>=0.4.2
torch>=2.0.0
numpy>=1.23.5
opencv-python>=4.11.0
PyYAML>=6.0.0
```

### Super Mario Bros环境

```
gymnasium==0.26.3
gym==0.23.1
gym-super-mario-bros==7.3.0
torch==2.7.0
numpy==1.23.5
opencv-python==4.11.0.86
PyYAML==6.0.2
nes-py==8.2.1
```

## 常见问题

### Atari 游戏相关问题

1. **找不到ALE命名空间**: 确保正确安装了gymnasium[atari]包并下载了ROM文件：
   ```bash
   pip install "gymnasium[atari]" ale-py "AutoROM[accept-rom-license]"
   autorom --accept-license
   ```

2. **内存不足错误**: 编辑config/breakout.yaml文件，减小REPLAY_BUFFER_CAPACITY和BATCH_SIZE。

3. **API版本不兼容错误**: 本项目已更新支持gymnasium v0.29+的新API。如果遇到解包错误，请检查您的gymnasium版本：
   ```bash
   pip show gymnasium
   ```

### Super Mario Bros相关问题

如果遇到numpy导入错误（例如循环导入问题），可以尝试以下解决方案：

```bash
# 激活虚拟环境
source .venv-mario/bin/activate

# 卸载并重新安装指定版本的numpy
pip uninstall -y numpy
pip install numpy==1.23.5

# 重新安装所有依赖
pip install -r requirements-mario.txt
```

如果问题仍然存在，可以尝试重建虚拟环境：

```bash
# 退出当前虚拟环境
deactivate

# 删除旧的虚拟环境
rm -rf .venv-mario

# 创建新的虚拟环境
python3 -m venv .venv-mario

# 激活环境
source .venv-mario/bin/activate

# 安装依赖
pip install -r requirements-mario.txt
```

## 参数说明

- `--game-type`: 选择游戏类型，可选值为`atari`或`mario`
- `--env`: Atari游戏环境名称，仅当game-type为atari时使用
- `--world-stage`: Super Mario Bros关卡，例如"1-1"表示第一世界第一关，仅当game-type为mario时使用
- `--algo`: 使用的算法，可选值为`random`或`dqn`
- `--mode`: 运行模式，可选值为`train`、`eval`或`train_and_eval`
- `--steps`: 训练步数，仅当mode为train或train_and_eval时使用
- `--render`: 是否渲染游戏画面，如果存在此参数则开启渲染
- `--episodes`: 评估时运行的episodes数量，默认为100
- `--record-video`: 是否录制游戏视频
- `--config`: 指定配置文件名称，例如"breakout"
- `--seed`: 随机种子 