#! /bin/bash

# 默认为离线模式
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 检查是否有 --online 参数
for arg in "$@"
do
    if [ "$arg" == "--online" ]; then
        unset HF_HUB_OFFLINE
        unset HF_DATASETS_OFFLINE
        echo "已切换到在线模式"
    fi
done

python -m gpt.run_experiments "$@"