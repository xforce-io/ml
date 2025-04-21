#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from log import INFO
import os
import sys

# 确保所有模块能被正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

from config import Config
from experiment import Experiment
from algos.random_algo import AlgoRandom
from algos.dqn_algo import AlgoDQN

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='强化学习Atari游戏实验')
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4', help='Atari游戏环境名称')
    parser.add_argument('--algo', type=str, default='random', choices=['random', 'dqn'], help='选择算法')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'train_and_eval'], help='运行模式')
    parser.add_argument('--steps', type=int, default=10000, help='训练总步数')
    parser.add_argument('--model', type=str, default=None, help='模型路径 (用于评估或加载预训练模型)')
    parser.add_argument('--episodes', type=int, default=10, help='评估的episodes数量')
    parser.add_argument('--render', action='store_true', help='开启环境渲染')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    parser.add_argument('--record-video', action='store_true', help='每10000步录制一个游戏视频')
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = Config()
    
    # 创建实验对象
    experiment = Experiment(
        env_name=args.env,
        config=config,
        render=args.render
    )
    
    # 打印运行环境信息
    INFO(logger, f"实验将在设备上运行: {experiment.device}")
    
    # 首先创建环境
    render_mode = "human" if args.render else None
    experiment.env = experiment._createEnv(render_mode=render_mode)
    
    # 然后选择算法
    if args.algo == 'random':
        algo = AlgoRandom(experiment.env, config, experiment.device)
    elif args.algo == 'dqn':
        algo = AlgoDQN(experiment.env, config, experiment.device)
    else:
        raise ValueError(f"不支持的算法: {args.algo}")
    
    # 设置算法
    experiment.setAlgo(algo)
    
    # 设置是否录制视频
    experiment.record_video = args.record_video
    
    # 如果提供了模型路径，尝试加载模型
    if args.model:
        model_path = args.model
        INFO(logger, f"加载模型: {model_path}")
    else:
        # 使用默认模型路径
        model_path = f"./saved_models/{args.algo}_{args.env}_final.pth"
        if args.mode in ['eval', 'train_and_eval']:
            INFO(logger, f"使用默认模型路径: {model_path}")
    
    # 根据模式运行实验
    if args.mode == 'train' or args.mode == 'train_and_eval':
        INFO(logger, f"训练 {args.algo} 算法在 {args.env} 环境")
        experiment.train(num_steps=args.steps)
        
        # 训练结束后，如果是'train_and_eval'模式，加载训练好的模型进行评估
        if args.mode == 'train_and_eval':
            INFO(logger, f"评估训练的模型: {model_path}")
            experiment.algo.load(model_path)
            experiment.evaluate(num_episodes=args.episodes)
    
    elif args.mode == 'eval':
        INFO(logger, f"评估 {args.algo} 算法在 {args.env} 环境")
        # 加载模型
        success = experiment.algo.load(model_path)
        if success:
            INFO(logger, f"评估加载的模型: {model_path}")
            experiment.evaluate(num_episodes=args.episodes)
        else:
            INFO(logger, f"无法加载模型: {model_path}，评估取消")
    
    # 如果开启了调试模式，展示Q值分布
    if args.debug and args.algo == 'dqn':
        INFO(logger, "\n===== 调试信息 =====")
        experiment.algo.visualizeQDistribution(num_samples=20)

if __name__ == "__main__":
    main() 