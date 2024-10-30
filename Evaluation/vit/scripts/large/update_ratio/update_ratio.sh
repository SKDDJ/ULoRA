#!/bin/bash
# Create a temporary file to store PIDs
pid_file="/tmp/task_pids_large_update_ratio"
# clear the last
rm /tmp/task_pids_large_update_ratio
echo "Starting Evaluations..."
date

MODEL_NAME="vit-l32-224-in21k" # Assuming a placeholder name for ViT/Large
NAME="large"

INIT_SARA_WEIGHTS="svd"
INIT_METHOD="svd"
LR_SCHEDULER="linear" 
OPT="adamw"
DATASETS=("cifar10" "cifar100" "dtd" "aircraft" "eurosat" "pets37" "flowers102" "cars")

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#AVAILABLE_GPUS[@]}
TASKS_PER_GPU=1

LR_DEFAULT=3e-4
LR_SPECIAL=3e-3

R=768
SEED=42
RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

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
    local ratio=$2
    local dataset=$3

    gpu_id=${AVAILABLE_GPUS[$gpu_index]}


    # Adjust LR if dataset is 'aircraft' or 'cars'
    if [[ "$dataset" == "aircraft" || "$dataset" == "cars" ]]; then
        LR=$LR_SPECIAL
    else
        LR=$LR_DEFAULT
    fi

    echo "Evaluating on ${dataset} with ${MODEL_NAME} and UPDATE Ratio ${ratio} on GPU $gpu_id..."
    CONFIG="/root/shiym_proj/Sara/vit-finetune/configs/sara/${dataset}.yaml"
    
    HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$gpu_id python /root/shiym_proj/Sara/vit-finetune/main.py fit --config $CONFIG --seed_everything $SEED --trainer.accelerator gpu --trainer.devices=[0] --trainer.precision 16-mixed \
        --trainer.max_epochs 10 --trainer.logger.name LARGE-Blora-Ratio-$ratio-R-$R-$NAME-$INIT_METHOD-$dataset-LR-$LR-SCHEDULER-$LR_SCHEDULER_NAME --model.warmup_steps 30 --model.lr $LR --model.scheduler $LR_SCHEDULER --model.optimizer $OPT \
        --data.batch_size 128 --data.dataset $dataset --data.workers 4 --model.model_name $MODEL_NAME \
        --model.lora_r $R --model.update_ratio $ratio --model.lora_alpha $R --model.init_method $INIT_METHOD --model.init_sara_weights $INIT_SARA_WEIGHTS > /root/shiym_proj/Sara/vit-finetune/logs/62-${INIT_METHOD}/vit_large_${ratio}_${dataset}.log 2>&1 &
    
    # Increment the task counter for the selected GPU
    gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] + 1))

    # Track the background job and GPU index for later management
    pid=$!
    echo "$pid $gpu_index" >> /tmp/task_pids_large_update_ratio
}


# Main training loop
for ratio in "${RATIOS[@]}"; do
    echo "Testing with ratio: $ratio"
    for DATASET in "${DATASETS[@]}"; do
        while true; do
            gpu_index=$(find_available_gpu)
            if [ $gpu_index -ne -1 ]; then
                run_task $gpu_index $ratio $DATASET
                sleep 30  # Wait before starting the next task
                break
            else
                sleep 1  # Wait before checking for available GPUs again
                # Check completed tasks and update gpu_tasks
                if [[ -s /tmp/task_pids_large_update_ratio ]]; then
                    while read -r pid gpu_index; do
                        if ! kill -0 $pid 2>/dev/null; then
                            gpu_tasks[$gpu_index]=$((gpu_tasks[$gpu_index] - 1))
                            sed -i "/^$pid $gpu_index$/d" /tmp/task_pids_large_update_ratio
                        fi
                    done < /tmp/task_pids_large_update_ratio
                fi
            fi
        done
    done
done

wait  # Wait for all tasks to complete
echo "All tasks completed, logs saved in vit_large_output.log"
date

# Cleanup after script finishes
cleanup