#!/bin/bash

# Conda 环境名称
CONDA_ENV_NAME="video-tools"

# 检查 conda 是否可用
if ! command -v conda &> /dev/null
then
    echo "Conda 未安装或未添加到 PATH"
    exit 1
fi

# 检查环境是否存在，不存在则创建
if ! conda env list | grep -q $CONDA_ENV_NAME; then
    echo "创建 Conda 环境 $CONDA_ENV_NAME"
    conda create -n $CONDA_ENV_NAME python=3.10 -y
fi

# 激活指定的 conda 环境（假设环境名为 whisper）
conda activate $CONDA_ENV_NAME


pip install git+https://github.com/m-bain/whisperx.git

# 检查并安装依赖
pip install -r requirements.txt

# 执行 Python 脚本
python main.py "$@"

# 取消激活环境（可选）
conda deactivate
