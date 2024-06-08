export WANDB_PROJECT=320-lr-dreambooth-sara

# 定义学习率数组
# big lr, we find 1.5e-2 to 5e-3 is good
# declare -a learning_rates=(5e-1 4.7e-1 4.4e-1 4.1e-1 3.8e-1 3.5e-1 3.2e-1 3e-1 2.8e-1 2.6e-1 2.4e-1 2.2e-1 2e-1 1.8e-1 1.6e-1 1.4e-1 1.2e-1 1e-1 8e-2 6e-2 4e-2 3e-2 2.5e-2 1.5e-2 5e-3)

# 定义学习率数组
# small lr
declare -a learning_rates=(1.5e-02 1.4e-02 1.3e-02 1.2e-02 1.1e-02 1.1e-02 1.0e-02 9.3e-03 8.8e-03 8.3e-03 7.8e-03 7.3e-03 6.9e-03 6.4e-03 6.1e-03 5.7e-03 5.4e-03 5.0e-03)

# 索引追踪学习率数组
index=0
total_lr=${#learning_rates[@]}

while true; do
    # 循环遍历每张显卡
    for ((gpu_id=0; gpu_id<8; gpu_id++)); do
        # 跳过5卡和6卡
        if [ $gpu_id -eq 5 ] || [ $gpu_id -eq 6 ]; then
            continue
        fi
        # 检查该卡是否有进程
        if ! pgrep -f "train_dreambooth.py --gpu_id $gpu_id" > /dev/null; then
            # 获取当前学习率
            learning_rate=${learning_rates[$index]}
            echo "Running on GPU $gpu_id with learning rate $learning_rate"

            export WANDB_NAME=lr-$learning_rate
            # 设置输出目录
            output_dir="/root/shiym_proj/Sara/peft/examples/lora_dreambooth/output/lr_${learning_rate//./}"

            # 启动训练任务
            CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch train_dreambooth.py \
                --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
                --instance_data_dir="./dog" \
                --output_dir="$output_dir" \
                --instance_prompt="a photo of sks dog" \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=1 \
                --checkpointing_steps=600 \
                --learning_rate=$learning_rate \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --max_train_steps=500 \
                --report_to="wandb" \
                --validation_prompt="A photo of sks dog in a bucket" \
                --num_validation_images=4 \
                --validation_steps=50 \
                --seed="42" \
                --use_lora \
                --no_tracemalloc &
            
            # 更新学习率索引
            index=$((index + 1))
            if [ $index -ge $total_lr ]; then
                echo "All learning rates have been used. Exiting."
                exit 0
            fi

            # 等待一秒钟，确保进程已经启动
            sleep 1
        fi
    done
    # 等待一段时间后再检查空闲GPU
    sleep 10
done







# export WANDB_MODE=offline
# export WANDB_PROJECT=lr-dreambooth-sara


# # 定义学习率数组
# # declare -a learning_rates=(2e-2 1e-2 5e-3 2.5e-3 1e-3 5e-4 2.5e-4 1e-4 5e-5)
# declare -a learning_rates=(5e-1 4.7e-1 4.4e-1 4.1e-1 3.8e-1 3.5e-1 3.2e-1 3e-1 2.8e-1 2.6e-1 2.4e-1 2.2e-1 2e-1 1.8e-1 1.6e-1 1.4e-1 1.2e-1 1e-1 8e-2 6e-2 4e-2 3e-2 2.5e-2 1.5e-2 5e-3)

# # 循环遍历每张显卡
# for ((gpu_id=0; gpu_id<8; gpu_id++)); do
#     # 跳过5卡和6卡
#     if [ $gpu_id -eq 5 ] || [ $gpu_id -eq 6 ]; then
#         continue
#     fi
#     # 检查该卡是否有进程
#     if ! pgrep -f "train_dreambooth.py --gpu_id $gpu_id" > /dev/null; then
#         # 获取当前学习率
#         learning_rate=${learning_rates[$gpu_id]}
#         echo "Running on GPU $gpu_id with learning rate $learning_rate"

#         export WANDB_NAME=lr-$learning_rate
#         # 设置输出目录
#         output_dir="/root/shiym_proj/Sara/peft/examples/lora_dreambooth/output/lr_$learning_rate"
        
#         # 启动训练任务
#         CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch train_dreambooth.py \
#             --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
#             --instance_data_dir="./dog" \
#             --output_dir="$output_dir" \
#             --instance_prompt="a photo of sks dog" \
#             --resolution=512 \
#             --train_batch_size=1 \
#             --gradient_accumulation_steps=1 \
#             --checkpointing_steps=600 \
#             --learning_rate=$learning_rate \
#             --lr_scheduler="constant" \
#             --lr_warmup_steps=0 \
#             --max_train_steps=500 \
#             --report_to="wandb" \
#             --validation_prompt="A photo of sks dog in a bucket" \
#             --num_validation_images=4 \
#             --validation_steps=50 \
#             --seed="42" \
#             --use_lora \
#             --no_tracemalloc &
        
#         # 等待一秒钟，确保进程已经启动
#         sleep 1
#     fi
# done
