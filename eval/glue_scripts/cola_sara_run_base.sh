#!/bin/bash

export HF_HOME=/root/.cache/huggingface
export XDG_CACHE_HOME=/root/.cache

# Assign epochs for cola
epochs_cola=80

# Different batch sizes you want to test
declare -A batch_sizes=(8 16 32 64)

# Different learning rates you want to test
declare -A learning_rates=("1e-2" "2e-3" "3e-4" "6e-5" "12e-6" "24e-7")

# Different scaling factors you want to test
declare -A lora_alpha=(8 16 80 128)
# 0.1 0.2 1.0 1.6

declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

export WANDB_MODE=offline

# SPECIFY THE DIRECTORIES
# exp_dir="/root/shiym_proj/Sara/logs/"

# Experiment variables
rank=8
l_num=12
seed=123

use_sara=True
train_classifier=True
target_modules="query value"
lora_dropout=0.
lora_bias=none
lora_task_type=SEQ_CLS

max_seq_length=512

export WANDB_PROJECT=cola-5-8-scaling_factor-bf16_sara_base_hp_LoRA

run(){
  learning_rate=$1
  batch_size=$2
  lora_alpha=$3
  task_name=$4

  export WANDB_NAME=base-${task_name}-r-${rank}-alpha-${lora_alpha}-target_modules-${target_modules}-seed-${seed}-bs-${batch_size}-lr-${learning_rate}-epochs-${epochs_cola}-with-weightdecay
  echo "Starting run ${WANDB_NAME}"

  HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes 2 --main_process_port 26686 ./run_glue_sara.py \
    --model_name_or_path FacebookAI/roberta-base \
    --task_name ${task_name} \
    --do_train --do_eval \
    --max_seq_length ${max_seq_length} \
    --per_device_train_batch_size ${batch_size} \
    --train_classifier ${train_classifier} \
    --use_sara ${use_sara} \
    --per_device_eval_batch_size ${batch_size} \
    --load_best_model_at_end True --metric_for_best_model ${metrics[$4]} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${epochs_cola} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --logging_steps 1 \
    --seed ${seed} \
    --lora_alpha ${lora_alpha} --lora_dropout 0. --lora_bias none \
    --target_modules ${target_modules} --rank ${rank} \
    --lora_task_type ${lora_task_type}  \
    --output_dir ${exp_dir}/model \
    --logging_dir ${exp_dir}/log \
    --overwrite_output_dir
  echo "task_name: ${task_name}, learning_rate: ${learning_rate}, batch_size: ${batch_size}, lora_alpha: ${lora_alpha}"
  echo "Finished run ${WANDB_NAME}"
}

# task_base=('mnli' 'qqp' 'cola' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' )
task_base=('cola' )

for task in "${task_base[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
      for alpha in "${lora_alpha[@]}"; do
        run  $lr $bs $alpha $task
      done
    done
  done
done


# #!/bin/bash

# export HF_HOME=/root/.cache/huggingface
# export XDG_CACHE_HOME=/root/.cache


# # LATEST
# # roberta base LoRA
# declare -A epochs=(["mnli"]=30 ["sst2"]=60 ["mrpc"]=30 ["cola"]=80 ["qnli"]=25 ["qqp"]=25 ["rte"]=80  ["stsb"]=40 )

# # roberta base LoRA
# declare -A bs=(["mnli"]=16 ["sst2"]=16 ["mrpc"]=16 ["cola"]=32 ["qnli"]=32 ["qqp"]=16 ["rte"]=32  ["stsb"]=16 )


# # roberta base LoRA
# declare -A ml=(["mnli"]=512 ["sst2"]=512 ["mrpc"]=512 ["cola"]=512 ["qnli"]=512 ["qqp"]=512 ["rte"]=512  ["stsb"]=512 )


# # roberta base LoRA
# declare -A lr=(["mnli"]="5e-4" ["sst2"]="5e-4" ["mrpc"]="4e-4" ["cola"]="4e-4" ["qnli"]="4e-4" ["qqp"]="5e-4" ["rte"]="5e-4"  ["stsb"]="4e-4" )

# declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

# # export WANDB_MODE=offline

# run(){
#   task_name=$1
#   learning_rate=${lr[$1]}
#   num_train_epochs=${epochs[$1]}
#   per_device_train_batch_size=${bs[$1]}
#   rank=8
#   l_num=12
#   seed=123
#   use_sara=True
#   train_classifier=True
#   lora_alpha=8
#   target_modules="query value"
#   lora_dropout=0.
#   lora_bias=none
#   lora_task_type=SEQ_CLS

#   export WANDB_PROJECT=5-8-scaling_factor_cola-bf16_sara_base_hp_LoRA
#   export WANDB_NAME=base-${task_name}-r-${rank}-target_modules-${target_modules}-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}

#   HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes 2 --main_process_port 26686 ./run_glue_sara.py \
#   --model_name_or_path FacebookAI/roberta-base  \
#   --task_name ${task_name} \
#   --do_train --do_eval \
#   --max_seq_length ${ml[$1]} \
#   --per_device_train_batch_size ${per_device_train_batch_size} \
#   --train_classifier ${train_classifier} \
#   --use_sara ${use_sara} \
#   --per_device_eval_batch_size ${per_device_train_batch_size} \
#   --load_best_model_at_end True --metric_for_best_model ${metrics[$1]} \
#   --learning_rate ${learning_rate} \
#   --num_train_epochs ${num_train_epochs} \
#   --evaluation_strategy epoch \
#   --save_strategy epoch \
#   --weight_decay 0. \
#   --warmup_ratio 0.06 \
#   --logging_steps 1 \
#   --seed ${seed} \
#   --lora_alpha ${lora_alpha} --lora_dropout 0. --lora_bias none \
#   --target_modules ${target_modules} --rank ${rank} \
#   --lora_task_type ${lora_task_type}  \
#   --output_dir ${exp_dir}/model \
#   --logging_dir ${exp_dir}/log \
#   --overwrite_output_dir
# }

# # task_base=('mnli' 'qqp' 'cola' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' )
# task_base=('cola'  )


# for task in "${task_base[@]}"; do
#     run $task
# done  