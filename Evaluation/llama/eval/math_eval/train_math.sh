#!/bin/bash

gpu=2

BASE_MODEL=Shiym/llama2-7B
OUTPUT=/home/yimingshi/shiym_proj/SaraEvaluation/llama/math_output
ADAPTER=/home/yimingshi/shiym_proj/SaraEvaluation/llama/math_output

R=1

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$gpu python train_math.py \
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
    --bf16 True \
    --tf32 True \
    --report_to none

CUDA_VISIBLE_DEVICES=$gpu python merge_adapter_to_base_model.py --base_mode $BASE_MODEL --adapter $ADAPTER/ft/ --output_path $OUTPUT --r $R --alpha $R
CUDA_VISIBLE_DEVICES=$gpu python inference/gsm8k_inference.py --model $OUTPUT
CUDA_VISIBLE_DEVICES=$gpu python inference/MATH_inference.py --model $OUTPUT