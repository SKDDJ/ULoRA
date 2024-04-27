#!/bin/bash

# 设置您想要的路径
PROCESSED_FOLDER_PATH="/root/shiym_proj/DiffLook/z_outputs"
OUTPUT_FOLDER_PATH="/root/shiym_proj/DiffLook/z_outputs"
# PROCESSED_FOLDER_PATH="/root/shiym_proj/DiffLook/predict/"
# OUTPUT_FOLDER_PATH="/root/shiym_proj/DiffLook/predict/"

REFERENCE_FIFE_PATH="/root/shiym_proj/DiffLook/utils/lora_reference_safetensors/zandaye.safetensors"

# 现在调用Python脚本，并传递这些路径作为参数
python utils/pt_to_safetensor.py "$PROCESSED_FOLDER_PATH" "$OUTPUT_FOLDER_PATH" "$REFERENCE_FIFE_PATH"
