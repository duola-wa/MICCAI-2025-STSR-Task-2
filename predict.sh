#!/bin/bash

# STSR Challenge 推理入口脚本
# 输入目录: /workspace/inputs/
# 输出目录: /workspace/outputs/

echo "=== STSR Challenge Inference Starting ==="
echo "Input directory: /workspace/inputs/"
echo "Output directory: /workspace/outputs/"

# 检查输入目录
if [ ! -d "/workspace/inputs" ]; then
    echo "Error: Input directory /workspace/inputs not found!"
    exit 1
fi

# 创建输出目录
mkdir -p /workspace/outputs

# 运行推理
echo "Running inference..."
cd /opt/app

python main_inference.py \
    --input_dir /workspace/inputs \
    --output_dir /workspace/outputs \
    --config_path inference_config.yaml

echo "=== STSR Challenge Inference Completed ==="