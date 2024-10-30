#!/bin/bash

gpu=6

BASE_MODEL=/root/shiym_proj/Sara/models/llama2_hf
OUTPUT=/root/shiym_proj/Sara/Evaluation/llama/math_output
ADAPTER=/root/shiym_proj/Sara/Evaluation/llama/math_output

R=1024

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$gpu python ../train_math.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT \
    --lora_r $R \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]"\
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
    --use_float fp16 
    # --use_gradient_checkpointing True \
    # The optimizer to use: adamw_torch, adamw_8bit, adamw_bnb_8bit, adamw_apex_fused, or adafactor
    # fp32, fp16, bf16
    # --fp16 True \ # toxic!
    # --tf32 False \

CUDA_VISIBLE_DEVICES=$gpu python ../merge_adapter_to_base_model.py --base_mode $BASE_MODEL --adapter $ADAPTER/ft/ --output_path $OUTPUT --r $R --alpha $R
CUDA_VISIBLE_DEVICES=$gpu python ../inference/gsm8k_inference.py --model $OUTPUT
CUDA_VISIBLE_DEVICES=$gpu python ../inference/MATH_inference.py --model $OUTPUT