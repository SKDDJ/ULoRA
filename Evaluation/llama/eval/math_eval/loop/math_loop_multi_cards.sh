#!/bin/bash

# 参数数组
FLOAT_TYPES=("fp16" "bf16")
RANKS=(256 512 1024 1536)

# 基础路径
BASE_PATH="/root/shiym_proj/Sara"
BASE_MODEL="$BASE_PATH/models/llama2_hf"

# 可用的 GPU
AVAILABLE_GPUS=(0 1 2 4 5 6 )
NUM_GPUS=${#AVAILABLE_GPUS[@]}
TASKS_PER_GPU=1

# 临时文件来存储 PID
PID_FILE="/tmp/llama_training_pids"

# 清除上次的 PID 文件
rm -f $PID_FILE

# 初始化数组来跟踪每个 GPU 上运行的任务
declare -a gpu_tasks
for ((i=0; i<$NUM_GPUS; i++)); do
    gpu_tasks[$i]=0
done

# 函数：查找可用的 GPU
find_available_gpu() {
    for ((i=0; i<$NUM_GPUS; i++)); do
        if [ ${gpu_tasks[$i]} -lt $TASKS_PER_GPU ]; then
            echo $i
            return
        fi
    done
    echo -1
}

# 清理函数
cleanup() {
    echo "正在停止所有后台进程..."
    if [[ -s $PID_FILE ]]; then
        while read -r pid gpu_index; do
            kill -TERM "$pid" 2>/dev/null
        done < "$PID_FILE"
    fi
    rm -f "$PID_FILE"
    exit 0
}

# 捕获 SIGINT (Ctrl+C) 并运行清理函数
trap cleanup SIGINT

# 运行单个训练任务的函数
run_task() {
    local gpu_index=$1
    local FLOAT=$2
    local R=$3
    local gpu_id=${AVAILABLE_GPUS[$gpu_index]}

    echo "开始运行 FLOAT=$FLOAT, R=$R 的实验，使用 GPU $gpu_id"

    # 设置输出和适配器路径
    OUTPUT="$BASE_PATH/Evaluation/llama/math_output/${R}_${FLOAT}"
    ADAPTER="$OUTPUT"

    # 创建日志文件
    LOG_FILE="${R}_${FLOAT}_gpu${gpu_id}.log"

    # 运行训练脚本
    HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$gpu_id python ../train_math.py \
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
    CUDA_VISIBLE_DEVICES=$gpu_id python ../merge_adapter_to_base_model.py \
        --base_mode $BASE_MODEL \
        --adapter $ADAPTER/ft/ \
        --output_path $OUTPUT \
        --r $R \
        --alpha $R \
        --use_float $FLOAT \
        >> "$LOG_FILE" 2>&1

    # 运行推理脚本
    CUDA_VISIBLE_DEVICES=$gpu_id python ../inference/gsm8k_inference.py --model $OUTPUT >> "$LOG_FILE" 2>&1
    CUDA_VISIBLE_DEVICES=$gpu_id python ../inference/MATH_inference.py --model $OUTPUT >> "$LOG_FILE" 2>&1

    echo "完成 FLOAT=$FLOAT, R=$R 的实验，使用 GPU $gpu_id"

    # 减少 GPU 任务计数
    gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] - 1))
    sed -i "/^$$ $gpu_index$/d" $PID_FILE
}

# 主训练循环
for FLOAT in "${FLOAT_TYPES[@]}"; do
    for R in "${RANKS[@]}"; do
        while true; do
            gpu_index=$(find_available_gpu)
            if [ $gpu_index -ne -1 ]; then
                run_task $gpu_index $FLOAT $R &
                pid=$!
                echo "$pid $gpu_index" >> $PID_FILE
                gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] + 1))
                sleep 30  # 等待一段时间再开始下一个任务
                break
            else
                sleep 5  # 等待一段时间再检查可用的 GPU
                # 检查已完成的任务并更新 gpu_tasks
                if [[ -s $PID_FILE ]]; then
                    while read -r pid gpu_index; do
                        if ! kill -0 $pid 2>/dev/null; then
                            gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] - 1))
                            sed -i "/^$pid $gpu_index$/d" $PID_FILE
                        fi
                    done < $PID_FILE
                fi
            fi
        done
    done
done

wait  # 等待所有任务完成
echo "所有实验已完成"

# 脚本结束时进行清理
cleanup