#!/bin/bash

# ================= 配置区域 =================

# Python 脚本的文件名 (请确保该文件在当前目录下)
PYTHON_SCRIPT="evaluation_search.py"

# 测试集数据路径 (这是我们在步骤2中生成的 benchmark 文件路径)
INPUT_FILE="./toolsciverifier/results_stage3/verifier_benchmark_dataset.jsonl"

# 结果保存目录
SAVE_DIR="./toolsciverifier/sciprm-bench/eval_res_search"

# 定义模型列表数组
# 请确保这些模型名称你的 API Server (127.0.0.1:3888) 支持
MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
)

# ================= 执行逻辑 =================

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 找不到 Python 脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 找不到测试数据集: $INPUT_FILE"
    echo "请先运行数据转换脚本生成测试集。"
    exit 1
fi

echo "🚀 开始批量评测任务..."
echo "📂 输入文件: $INPUT_FILE"
echo "📂 结果目录: $SAVE_DIR"
echo "--------------------------------------------------------"

# 循环遍历模型并顺序执行
for model in "${MODELS[@]}"; do
    echo "========================================================"
    echo "🤖 正在评测模型: $model"
    echo "========================================================"
    
    # 执行 Python 脚本
    python "$PYTHON_SCRIPT" \
      --model "$model" \
      --input-file "$INPUT_FILE" \
      --save-dir "$SAVE_DIR" \
      --max-workers 32
    
    # 获取上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✅ 模型 $model 评测完成"
    else
        echo "❌ 模型 $model 评测过程中发生错误"
        # 如果希望遇到错误停止整个流程，请取消下面一行的注释
        # exit 1 
    fi
    
    echo ""
    echo "--------------------------------------------------------"
    echo "⏳ 冷却 5 秒..."
    sleep 5
done

echo "🎉 所有模型评测任务结束！结果已保存在 $SAVE_DIR"
