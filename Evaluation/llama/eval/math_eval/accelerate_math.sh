#!/bin/bash

BASE_MODEL=/root/shiym_proj/Sara/models/llama2_hf

R=1536
FLOAT=fp16

# export WANDB_MODE=offline

BASE_PATH="/root/shiym_proj/Sara"
# 设置输出和适配器路径
OUTPUT="$BASE_PATH/Evaluation/llama/math_output/${R}_${FLOAT}"
ADAPTER="$OUTPUT"

# OUTPUT=/root/shiym_proj/Sara/Evaluation/llama/math_output/$R_$FLOAT
# ADAPTER=/root/shiym_proj/Sara/Evaluation/llama/math_output$R_$FLOAT

    accelerate launch --mixed_precision fp16 train_math_accelerate.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT \
    --lora_r $R \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]"\
    --dataset_field query response \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --optim adamw_torch \
    --use_float $FLOAT \
    # --max_steps 1 
    # --save_steps 100 \
    # --use_gradient_checkpointing True \
    # The optimizer to use: adamw_torch, adamw_8bit, adamw_bnb_8bit, adamw_apex_fused, or adafactor
    # fp32, fp16, bf16

# CUDA_VISIBLE_DEVICES=$gpu python merge_adapter_to_base_model.py --base_mode $BASE_MODEL --adapter $ADAPTER/ft/ --output_path $OUTPUT --r $R --alpha $R --use_float $FLOAT
# CUDA_VISIBLE_DEVICES=$gpu python inference/gsm8k_inference.py --model $OUTPUT
# CUDA_VISIBLE_DEVICES=$gpu python inference/MATH_inference.py --model $OUTPUT


# python /root/shiym_proj/Sara/Evaluation/llama/eval/math_eval/inference/gsm8k_inference.py --model /root/shiym_proj/Sara/Evaluation/llama/math_output/1536_fp16/Trained_llama --data_path /root/shiym_proj/Sara/Evaluation/llama/eval/math_eval/data/gsm8k_test.jsonl

# python /root/shiym_proj/Sara/Evaluation/llama/eval/math_eval/inference/MATH_inference.py --model /root/shiym_proj/Sara/Evaluation/llama/math_output/1536_fp16/Trained_llama --data_path /root/shiym_proj/Sara/Evaluation/llama/eval/math_eval/data/MATH_test.jsonl

