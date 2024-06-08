# export WANDB_MODE=offline
export WANDB_PROJECT=320-girl-1.5-adamw-attn-bcz

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export VAE_NAME="/root/.cache/huggingface/hub/sd-vae-ft-mse"
# export MODEL_NAME="SG161222/Realistic_Vision_V6.0_B1_noVAE"
# 定义学习率数组
# big lr, we find 1.5e-2 to 5e-3 is good
# declare -a learning_rates=(5e-1 4.7e-1 4.4e-1 4.1e-1 3.8e-1 3.5e-1 3.2e-1 3e-1 2.8e-1 2.6e-1 2.4e-1 2.2e-1 2e-1 1.8e-1 1.6e-1 1.4e-1 1.2e-1 1e-1 8e-2 6e-2 4e-2 3e-2 2.5e-2 1.5e-2 5e-3)
# small lr
# declare -a learning_rates=(2.4e-02 2.5e-02 2.8e-02 2.9e-02 3.1e-02 3.2e-02)
# declare -a learning_rates=(5e-03 1.6e-02 1.7e-02 1.8e-02 1.9e-02 2e-02 2.1e-02 2.2e-02 2.3e-02  2.6e-02 2.7e-02  3.3e-02 3.4e-02 3.5e-02 6e-2 4e-2 2.4e-1 2.2e-1)
# declare -a learning_rates=(1)
# 索引追踪学习率数组
# index=0
# total_lr=${#learning_rates[@]}

# while true; do
#     # 循环遍历每张显卡
#     for ((gpu_id=0; gpu_id<8; gpu_id++)); do
#         # 跳过5卡和6卡
#         if [ $gpu_id -eq 0 ] || [ $gpu_id -eq 1 ] || [ $gpu_id -eq 2 ]|| [ $gpu_id -eq 6 ]|| [ $gpu_id -eq 7 ]; then
#             continue
#         fi
        # 检查该卡是否有进程
        # if ! pgrep -f "train_dreambooth.py" > /dev/null; then
# 获取当前学习率
# learning_rate=${learning_rates[$index]}
# echo "Running on GPU $gpu_id with learning rate $learning_rate"

precision="fp16"
export WANDB_NAME=adamw-bcz4-lr1e-1-worker4-crossattn-only
# 设置输出目录
output_dir="/home/yimingshi/shiym_proj/Sarapeft/examples/lora_dreambooth/output/$precision-15-prodigy"
# gpu_id=0
# gpu_id=1
# gpu_id=3
gpu_id=5
# 启动训练任务
CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch --mixed_precision=$precision train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --instance_data_dir="./single_woman_file/woman" \
    --output_dir="$output_dir" \
    --instance_prompt="a photo of sks girl" \
    --resolution=512 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=12000 \
    --optimizer="adamw" \
    --learning_rate=1e-1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=15 \
    --max_train_steps=10000 \
    --report_to="wandb" \
    --num_validation_images=2 \
    --validation_prompt="a photo of sks girl, detailed face, highres, RAW photo 8k uhd" \
    --validation_steps=1000 \
    --seed="42" \
    --use_lora \
    --no_tracemalloc \
    --adam_weight_decay=0.01 \
    --center_crop \
    --num_dataloader_workers=4 \
    --lora_target_modules="attn2.to_k","attn2.to_v","attn2.to_q","attn2.to_out" \
    # --adam_beta2=0.99 \


    # --snr_gamma=5 \




# # test only self attn
# --lora_target_modules="attn1.to_k","attn1.to_v","attn1.to_q","attn1.to_out" \

# # test only cross attn
# --lora_target_modules="attn2.to_k","attn2.to_v","attn2.to_q","attn2.to_out" \

# # test both self and cross attn
# --lora_target_modules="attn1.to_k","attn1.to_v","attn1.to_q","attn1.to_out","attn2.to_k","attn2.to_v","attn2.to_q","attn2.to_out" \



# --prodigy_decouple \
#     --prodigy_use_bias_correction \
#     --prodigy_safeguard_warmup \

    # --enable_xformers_memory_efficient_attention



#             # 更新学习率索引
#             index=$((index + 1))
#             if [ $index -ge $total_lr ]; then
#                 echo "All learning rates have been used. Exiting."
#                 exit 0
#             fi

#             # 等待一秒钟，确保进程已经启动
#             sleep 1
#         fi
#     done
#     # 等待一段时间后再检查空闲GPU
#     sleep 10
# done

# --validation_prompt="A photo of girl, detailed face, highres, RAW photo 8k uhd" \˛
# --validation_prompt="A photo of girl with green hair, detailed face, highres, RAW photo 8k uhd" \˛
# --validation_prompt="masterpiece, best quality,(detailed face, perfect face, perfect eyes, realistic eyes, perfect fingers),(clear face),fantasy girl,long hair,hair ornaments,looking at viewer,outdoors,intricate,high detail,sharp focus,dramatic,beautiful girl,full body,kneeling, outdoors, white dress,bow,cleavage." \
    # --validation_prompt="masterpiece, best quality, (detailed face, perfect eyes, realistic eyes, perfect fingers), fantasy girl with long hair and moon ornaments, outdoors at night in a star-patterned blue gown, (clear face), full body, under a starry sky.  " \