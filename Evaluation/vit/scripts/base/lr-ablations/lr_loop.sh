#!/bin/bash

# Create a temporary file to store PIDs
pid_file="/tmp/task_pids_base_lr"
# clear the last
rm /tmp/task_pids_base_lr
echo "Starting Evaluations..."
date

MODEL_NAME="vit-b16-224-in21k"
NAME="base"
INIT_SARA_WEIGHTS="ldu"
INIT_METHOD="lu"
LR_SCHEDULER="linear" 
OPT="adamw"
DATASETS=("cifar10" "cifar100" "dtd" "aircraft" "eurosat" "pets37" "flowers102" "cars")

AVAILABLE_GPUS=(2 4 5)
NUM_GPUS=${#AVAILABLE_GPUS[@]}
TASKS_PER_GPU=2

R=768
ALPHA=768
FULLRANK="768-1"
SEED=42
LEARNING_RATES=(1e-1 5e-2 8e-3 5e-3 3e-3 6e-4 3e-4 1e-5)

# Initialize array to track running tasks per GPU
declare -a gpu_tasks
for ((i=0; i<$NUM_GPUS; i++)); do
    gpu_tasks[$i]=0
done

# Function to find an available GPU
find_available_gpu() {
    for ((i=0; i<$NUM_GPUS; i++)); do
        if [ ${gpu_tasks[$i]} -lt $TASKS_PER_GPU ]; then
            echo $i
            return
        fi
    done
    echo -1
}

run_task() {
    local gpu_index=$1
    local lr=$2
    local dataset=$3

    gpu_id=${AVAILABLE_GPUS[$gpu_index]}
    echo "Evaluating on ${dataset} with ${MODEL_NAME} and learning rate ${lr} on GPU $gpu_id..."
    CONFIG="/root/shiym_proj/Sara/vit-finetune/configs/sara/${dataset}.yaml"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python /root/shiym_proj/Sara/vit-finetune/main.py fit --config $CONFIG --seed_everything $SEED --trainer.accelerator gpu --trainer.devices=[0] --trainer.precision 16-mixed \
        --trainer.max_epochs 10 --trainer.logger.name TESTLR-$lr-$NAME-$INIT_METHOD-$FULLRANK-$dataset-R-$R-SCHEDULER-$LR_SCHEDULER_NAME --model.warmup_steps 30 --model.lr $lr --model.scheduler $LR_SCHEDULER --model.optimizer $OPT \
        --data.batch_size 128 --data.dataset $dataset --data.workers 4 --model.model_name $MODEL_NAME \
        --model.lora_r $R --model.lora_alpha $ALPHA --model.init_method $INIT_METHOD --model.init_sara_weights $INIT_SARA_WEIGHTS > /root/shiym_proj/Sara/vit-finetune/logs/61-${INIT_METHOD}/vit_large_${lr}_${dataset}.log 2>&1 &
    
    sleep 5  # Wait before starting the next task
    # Increment the task counter for the selected GPU
    gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] + 1))

    # Track the background job and GPU index for later management
    pid=$!
    echo "$pid $gpu_index" >> /tmp/task_pids_base_lr
}

# Main training loop
for LR in "${LEARNING_RATES[@]}"; do
    echo "Testing with learning rate: $LR"
    for DATASET in "${DATASETS[@]}"; do
        while true; do
            gpu_index=$(find_available_gpu)
            if [ $gpu_index -ne -1 ]; then
                run_task $gpu_index $LR $DATASET
                break
            else
                sleep 1  # Wait before checking for available GPUs again
                # Check completed tasks and update gpu_tasks
                if [[ -s /tmp/task_pids_base_lr ]]; then
                    while read -r pid gpu_index; do
                        if ! kill -0 $pid 2>/dev/null; then
                            gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] - 1))
                            sed -i "/^$pid $gpu_index$/d" /tmp/task_pids_base_lr
                        fi
                    done < /tmp/task_pids_base_lr
                fi
            fi
        done
    done
done


wait  # Wait for all tasks to complete
echo "All tasks completed, logs saved in vit_base_output.log"
date
