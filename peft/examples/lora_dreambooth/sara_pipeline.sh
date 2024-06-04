#!/bin/bash
export WANDB_PROJECT=320-woman-pipeline-maker


# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="SG161222/Realistic_Vision_V6.0_B1_noVAE"
# export VAE_NAME="stabilityai/sd-vae-ft-mse-original"

# 定义固定学习率
learning_rate=1.0

# 已处理的ID记录文件
processed_ids_file="processed_ids.txt"
touch $processed_ids_file

# 读取已处理的ID到数组中
processed_ids=()
if [ -f "$processed_ids_file" ]; then
    mapfile -t processed_ids < $processed_ids_file
fi

# 定义数据目录
data_dir="/root/shiym_proj/DiffLook/outputs/"

# 获取所有人物ID列表
ids=()
for part_dir in "$data_dir"/part*; do
    for id_dir in "$part_dir"/*; do
        id=$(basename "$id_dir")
        if [[ ! " ${processed_ids[*]} " =~ " ${id} " ]]; then
            ids+=("$id_dir")
        fi
    done
done

while true; do
    for ((gpu_id=0; gpu_id<8; gpu_id++)); do
        # if [ $gpu_id -eq 5 ] || [ $gpu_id -eq 6 ]; then
        #     continue
        # fi
        if ![ $gpu_id -eq 3 ] ; then
            continue
        fi
        
        if ! pgrep -f "train_dreambooth.py" > /dev/null; then
            if [ ${#ids[@]} -eq 0 ]; then
                echo "All IDs have been processed. Exiting."
                exit 0
            fi

            id_dir="${ids[0]}"
            id=$(basename "$id_dir")

            echo "Running on GPU $gpu_id for ID $id"

            export WANDB_NAME=$id
            output_dir="/root/shiym_proj/Sara/peft/examples/lora_dreambooth/output/${id}"

            CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch train_dreambooth.py \
                --pretrained_model_name_or_path=$MODEL_NAME  \
                --instance_data_dir="$id_dir" \
                --output_dir="$output_dir" \
                --instance_prompt="a photo of woman" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --checkpointing_steps=700 \
                --learning_rate=$learning_rate \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --max_train_steps=500 \
                --validation_prompt="A photo of woman" \
                --num_validation_images=4 \
                --validation_steps=50 \
                --seed="42" \
                --use_lora \
                --no_tracemalloc \
                --report_to="wandb" &

            echo "$id" >> $processed_ids_file
            ids=("${ids[@]:1}")

            sleep 1
        fi
    done
    sleep 10
done
