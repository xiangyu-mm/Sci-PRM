#!/bin/bash

# 定义模型列表数组
# 注意：这里根据常见的 HuggingFace 格式调整了部分大小写，如果你的模型是本地路径，请修改为实际路径
MODELS=(
    "qwen3-vl-8b-instruct"
)

# 循环遍历模型并顺序执行
for model in "${MODELS[@]}"; do
    echo "========================================================"
    echo "正在开始评测模型: $model"
    echo "========================================================"
    
    python evaluation_benchmark.py \
      --model "$model" \
      --max-workers 32
    
    # 获取上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✅ 模型 $model 评测完成"
    else
        echo "❌ 模型 $model 评测失败"
        # 如果你想遇到错误就停止整个脚本，取消下面这行的注释
        # exit 1 
    fi
    
    echo ""
    echo "--------------------------------------------------------"
    # 可选：休息几秒让 GPU 冷却或清理
    sleep 5
done

echo "🎉 所有模型评测任务结束！"
