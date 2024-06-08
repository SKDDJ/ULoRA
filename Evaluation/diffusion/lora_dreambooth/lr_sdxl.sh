

# export WANDB_MODE=offline
export WANDB_PROJECT=xl-dreambooth-sara
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./dog_file/dog"
# export OUTPUT_DIR="output-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"


# 定义学习率数组
# big lr, we find 1.5e-2 to 5e-3 is good
# declare -a learning_rates=(5e-1 4.7e-1 4.4e-1 4.1e-1 3.8e-1 3.5e-1 3.2e-1 3e-1 2.8e-1 2.6e-1 2.4e-1 2.2e-1 2e-1 1.8e-1 1.6e-1 1.4e-1 1.2e-1 1e-1 8e-2 6e-2 4e-2 3e-2 2.5e-2 1.5e-2 5e-3)

# 定义学习率数组
# small lr
# declare -a learning_rates=(2.4e-02 2.5e-02 2.8e-02 2.9e-02 3.1e-02 3.2e-02)
declare -a learning_rates=()
# 索引追踪学习率数组
index=0
total_lr=${#learning_rates[@]}

while true; do
    # 循环遍历每张显卡
    for ((gpu_id=0; gpu_id<8; gpu_id++)); do
        # 跳过5卡和6卡
        # if [ $gpu_id -eq 5 ] || [ $gpu_id -eq 6 ] || [ $gpu_id -eq 1 ] || [ $gpu_id -eq 0 ]; then
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
            output_dir="/home/yimingshi/shiym_proj/Sarapeft/examples/lora_dreambooth/output-xl/lr_${learning_rate//./}"

            # 启动训练任务
            CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch train_dreambooth_sdxl.py \
            --pretrained_model_name_or_path=$MODEL_NAME  \
            --instance_data_dir=$INSTANCE_DIR \
            --pretrained_vae_model_name_or_path=$VAE_PATH \
            --output_dir=$output_dir \
            --mixed_precision="bf16" \
            --instance_prompt="a photo of dog" \
            --resolution=1024 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=4 \
            --learning_rate=1e-4 \
            --report_to="wandb" \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps=500 \
            --validation_prompt="A photo of dog in a bucket" \
            --validation_epochs=25 \
            --seed="42" \
            --use_lora &
            
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

