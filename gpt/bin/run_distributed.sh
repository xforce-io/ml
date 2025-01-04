#!/bin/bash

# 默认配置
NUM_GPUS=4  # 默认使用4个GPU
MASTER_PORT=29500  # 默认端口号
GPU_INDICES="0,1,2,3"  # 默认使用的GPU索引

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu_indices)
            GPU_INDICES="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=$GPU_INDICES

# 打印配置信息
echo "启动分布式训练："
echo "- 使用 GPU 数量: $NUM_GPUS"
echo "- 使用 GPU 索引: $GPU_INDICES"
echo "- 主端口: $MASTER_PORT"

# 使用 torch.distributed.launch 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m gpt.run_experiments