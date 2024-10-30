#!/bin/bash

# 定义参数数组
FLOAT_TYPES=("fp16" "bf16")
RANKS=(256 512 1024 1536)

# 基础路径
BASE_PATH="/root/shiym_proj/Sara"
BASE_MODEL="$BASE_PATH/models/llama2_hf"

# 循环遍历所有组合
for FLOAT in "${FLOAT_TYPES[@]}"; do
    for R in "${RANKS[@]}"; do
        echo "Starting run with FLOAT=$FLOAT and R=$R"
        
        # 设置输出和适配器路径
        OUTPUT="$BASE_PATH/Evaluation/llama/math_output/${R}_${FLOAT}"
        ADAPTER="$OUTPUT"
        
        # 创建日志文件
        LOG_FILE="${R}_${FLOAT}.log"
        
        # 运行训练脚本
        CUDA_VISIBLE_DEVICES=0 python ../train_math.py \
            --model_name_or_path $BASE_MODEL \
            --output_dir $OUTPUT \
            --lora_r $R \
            --data_path meta-math/MetaMathQA \
            --dataset_split "train[:100000]" \
            --dataset_field query response \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 128 \
            --save_strategy "steps" \
            --save_steps 100 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --report_to wandb \
            --optim adamw_torch \
            --use_float $FLOAT \
            >> "$LOG_FILE" 2>&1
        
        # 运行合并脚本
        CUDA_VISIBLE_DEVICES=0 python ../merge_adapter_to_base_model.py \
            --base_mode $BASE_MODEL \
            --adapter $ADAPTER/ft/ \
            --output_path $OUTPUT \
            --r $R \
            --alpha $R \
            --use_float $FLOAT \
            >> "$LOG_FILE" 2>&1
        
        # 运行推理脚本
        CUDA_VISIBLE_DEVICES=0 python ../inference/gsm8k_inference.py --model $OUTPUT >> "$LOG_FILE" 2>&1
        CUDA_VISIBLE_DEVICES=0 python ../inference/MATH_inference.py --model $OUTPUT >> "$LOG_FILE" 2>&1
        
        echo "Finished run with FLOAT=$FLOAT and R=$R"
    done
done

echo "All runs completed."