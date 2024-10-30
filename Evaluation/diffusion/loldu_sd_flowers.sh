#!/bin/bash

export WANDB_PROJECT=new-727-loldu-diffusion
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export VAE_NAME="/root/.cache/huggingface/hub/sd-vae-ft-mse"

declare -a learning_rates=(5e-4)

index=0
total_lr=${#learning_rates[@]}

while true; do
    for ((gpu_id=0; gpu_id<8; gpu_id++)); do
        if [ $gpu_id -ne 4 ]; then
            continue
        fi

            learning_rate=${learning_rates[$index]}
            echo "Running on GPU $gpu_id with learning rate $learning_rate"

            precision="fp16"
            export WANDB_NAME=flowers-loldu-diffusion-$learning_rate
            output_dir="/root/shiym_proj/Sara/Evaluation/diffusion/lora_dreambooth/outputs/flowers/$precision-727"

            CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch --mixed_precision=$precision train_dreambooth.py \
                --pretrained_model_name_or_path=$MODEL_NAME \
                --pretrained_vae_model_name_or_path=$VAE_NAME \
                --instance_data_dir="./datasets/flowers" \
                --output_dir="$output_dir" \
                --instance_prompt="a photo of sks flower" \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=1 \
                --checkpointing_steps=10000 \
                --optimizer="adamw" \
                --learning_rate=$learning_rate \
                --lr_scheduler="constant" \
                --lr_warmup_steps=15 \
                --max_train_steps=1000 \
                --report_to="wandb" \
                --num_validation_images=4 \
                --validation_prompt="a photo of sks flower flying in the sky" \
                --validation_steps=100 \
                --seed="42" \
                --use_lora \
                --no_tracemalloc \
                --adam_weight_decay=0.01 \
                --center_crop \
                --num_dataloader_workers=1 \
                --lora_target_modules="to_k","to_v","to_q","to_out"

            index=$((index + 1))
            if [ $index -eq $total_lr ]; then
                index=0
            fi
    done
    sleep 60
done