#!/bin/bash

# export WANDB_MODE=offline
export HF_HOME=/root/.cache/huggingface
export XDG_CACHE_HOME=/root/.cache

# Assign epochs for cola
epochs=30

# Different batch sizes you want to test
# declare -A batch_sizes=( [8]=8 [16]=16 [32]=32 [64]=64 )
declare -A batch_sizes=([64]=64) # fourier FT all bcz of base is 32

# Different learning rates you want to test
# declare -A learning_rates=(["5e-2"]="5e-2" ["2e-2"]="2e-2" ["5e-3"]="5e-3" ["1e-3"]="1e-3" ["5e-4"]="5e-4" ["3e-5"="3e-5"] )
# declare -A learning_rates=(["1e-3"]="1e-3" ["5e-4"]="5e-4" ["3e-5"="3e-5"] )
declare -A learning_rates=(["1"]="1")
# declare -A learning_rates=(["3e-5"="3e-5"] )

# Different scaling factors you want to test
declare -A lora_alpha=( [768]=768 )

declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

# SPECIFY THE DIRECTORIES
exp_dir="/root/shiym_proj/Sara/logs/"

# Experiment variables
rank=768
seed=42

use_sara=True
train_classifier=True
target_modules="query value key output.dense"
lora_task_type=SEQ_CLS
max_seq_length=512

export WANDB_PROJECT=LoLDU-CoLA

run(){
  learning_rate=$1
  batch_size=$2
  lora_alpha=$3
  task_name=$4

  export WANDB_NAME=lr-${learning_rate}-r-${rank}-alpha-${lora_alpha}-target_modules-${target_modules}-seed-${seed}-bs-${batch_size}-epochs-${epochs}
  echo "Starting run ${WANDB_NAME}"

  HF_ENDPOINT=https://hf-mirror.com accelerate launch --mixed_precision 'bf16' --num_processes 2 --main_process_port 26686 ./run_glue_sara.py \
    --model_name_or_path FacebookAI/roberta-base \
    --task_name ${task_name} \
    --do_train --do_eval \
    --max_seq_length ${max_seq_length} \
    --per_device_train_batch_size ${batch_size} \
    --train_classifier ${train_classifier} \
    --use_sara ${use_sara} \
    --per_device_eval_batch_size ${batch_size} \
    --load_best_model_at_end True --metric_for_best_model ${metrics[$task_name]} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${epochs} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --seed ${seed} \
    --lora_alpha ${lora_alpha} --lora_dropout 0. --lora_bias none \
    --target_modules ${target_modules} --rank ${rank} \
    --lora_task_type ${lora_task_type}  \
    --output_dir ${exp_dir}/model \
    --logging_dir ${exp_dir}/log \
    --overwrite_output_dir
    # --weight_decay 0.01 \
  echo "task_name: ${task_name}, learning_rate: ${learning_rate}, batch_size: ${batch_size}, lora_alpha: ${lora_alpha}"
  echo "Finished run ${WANDB_NAME}"
}

# task_base=('cola')

# task_base=('cola' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' )
# task_base=('qnli' 'rte' 'sst2' 'stsb' )
task_base=('mnli' 'qqp' )
# task_base=('sst2' 'stsb' )

for task in "${task_base[@]}"; do
  for lr in "${!learning_rates[@]}"; do
    for bs in "${!batch_sizes[@]}"; do
      for alpha in "${!lora_alpha[@]}"; do
        run  ${learning_rates[$lr]} ${batch_sizes[$bs]} ${lora_alpha[$alpha]} $task
      done
    done
  done
done
