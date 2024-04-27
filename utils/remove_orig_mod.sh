#!/bin/bash

# 设置您想要的路径
COMPILED_CKPT_FOLDER_PATH="/root/shiym_proj/DiffLook/epoch_099.ckpt"

# 现在调用Python脚本，并传递这些路径作为参数
python utils/trans_compiled_model.py "$COMPILED_CKPT_FOLDER_PATH"
