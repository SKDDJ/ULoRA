#!/bin/bash
# clear the last
rm /tmp/task_pids_large_rank
echo "Starting Evaluations..."
date

MODEL_NAME="vit-l32-224-in21k" # Assuming a placeholder name for ViT/Large
NAME="large"
INIT_SARA_WEIGHTS="ldu"
INIT_METHOD="lu"
LR_SCHEDULER="linear" 
OPT="adamw"
DATASETS=("cifar10" "cifar100" "dtd" "aircraft" "eurosat" "pets37" "flowers102" "cars")

AVAILABLE_GPUS=(0 1 3 6 7)
NUM_GPUS=${#AVAILABLE_GPUS[@]}
TASKS_PER_GPU=2

LR=5e-3

SEED=42
RANKS=(1 8 16 32 64 128 256 512 768)

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

# Create a temporary file to store PIDs
pid_file="/tmp/task_pids_large_rank"

# Cleanup function to kill all background processes
cleanup() {
    echo "Stopping all background processes..."
    if [[ -s $pid_file ]]; then
        while read -r pid gpu_index; do
            kill -TERM "$pid" 2>/dev/null
        done < "$pid_file"
    fi
    rm -f "$pid_file"
    exit 0
}

# Trap SIGINT (Ctrl+C) and run the cleanup function
trap cleanup SIGINT

# Function to run a single training task
run_task() {
    local gpu_index=$1
    local R=$2
    local dataset=$3

    gpu_id=${AVAILABLE_GPUS[$gpu_index]}
    echo "Evaluating on ${dataset} with ${MODEL_NAME} and Rank ${R} on GPU $gpu_id..."
    CONFIG="/root/shiym_proj/Sara/vit-finetune/configs/sara/${dataset}.yaml"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python /root/shiym_proj/Sara/vit-finetune/main.py fit --config $CONFIG --seed_everything $SEED --trainer.accelerator gpu --trainer.devices=[0] --trainer.precision 16-mixed \
        --trainer.max_epochs 10 --trainer.logger.name RANKRANK-$R-$NAME-$INIT_METHOD-$dataset-LR-$LR-SCHEDULER-$LR_SCHEDULER_NAME --model.warmup_steps 30 --model.lr $LR --model.scheduler $LR_SCHEDULER --model.optimizer $OPT \
        --data.batch_size 128 --data.dataset $dataset --data.workers 4 --model.model_name $MODEL_NAME \
        --model.lora_r $R --model.lora_alpha $R --model.init_method $INIT_METHOD --model.init_sara_weights $INIT_SARA_WEIGHTS > /root/shiym_proj/Sara/vit-finetune/logs/62-${INIT_METHOD}/vit_large_${R}_${dataset}.log 2>&1 &
    
    sleep 5  # Wait before starting the next task
    # Increment the task counter for the selected GPU
    gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] + 1))

    # Track the background job and GPU index for later management
    pid=$!
    echo "$pid $gpu_index" >> /tmp/task_pids_large_rank
}


# Main training loop
for R in "${RANKS[@]}"; do
    echo "Testing with rank: $R"
    for DATASET in "${DATASETS[@]}"; do
        while true; do
            gpu_index=$(find_available_gpu)
            if [ $gpu_index -ne -1 ]; then
                run_task $gpu_index $R $DATASET
                break
            else
                sleep 1  # Wait before checking for available GPUs again
                # Check completed tasks and update gpu_tasks
                if [[ -s /tmp/task_pids_large_rank ]]; then
                    while read -r pid gpu_index; do
                        if ! kill -0 $pid 2>/dev/null; then
                            gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] - 1))
                            sed -i "/^$pid $gpu_index$/d" /tmp/task_pids_large_rank
                        fi
                    done < /tmp/task_pids_large_rank
                fi
            fi
        done
    done
done

wait  # Wait for all tasks to complete
echo "All tasks completed, logs saved in vit_base_output.log"
date

# Cleanup after script finishes
cleanup