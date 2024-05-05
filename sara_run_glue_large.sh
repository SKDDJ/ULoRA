#!/bin/bash

export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
export HF_HOME=/root/.cache/huggingface
export XDG_CACHE_HOME=/root/.cache

# roberta large
declare -A epochs=(["mnli"]=10 ["mrpc"]=40 ["qnli"]=20 ["qqp"]=10 ["rte"]=40 ["sst2"]=10 ["stsb"]=20 ["cola"]=40)


# roberta large
declare -A bs=(["mnli"]=32 ["mrpc"]=32 ["qnli"]=32 ["qqp"]=32 ["rte"]=32 ["sst2"]=32 ["stsb"]=32 ["cola"]=32)


# roberta large VeRA
declare -A ml=(["mnli"]=128 ["mrpc"]=128 ["qnli"]=128 ["qqp"]=128 ["rte"]=128 ["sst2"]=128 ["stsb"]=128 ["cola"]=128)

# roberta large 
declare -A lr=(["mnli"]="1e-2" ["mrpc"]="3e-2" ["qnli"]="1e-2" ["qqp"]="1e-2" ["rte"]="2e-2" ["sst2"]="1e-2" ["stsb"]="2e-2" ["cola"]="1e-2")

declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

# export WANDB_MODE=offline
seed=42

run(){
  task_name=$1
  learning_rate=${lr[$1]}
  num_train_epochs=${epochs[$1]}
  per_device_train_batch_size=${bs[$1]}
  rank=1024
  l_num=12
  seed=42
  use_sara=True
  train_classifier=True
  lora_alpha="1024"
  target_modules="query value key"
  mode=$4
  lora_dropout=0.
  lora_bias=none
  lora_task_type=SEQ_CLS
  wandb_project=sara_large_hp_vera_glue
  wandb_run_name=roberta-large-sara-${task_name}-r-${rank}-qkv--seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}

  HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes 4 --main_process_port 26688 ./run_glue_sara.py \
  --model_name_or_path FacebookAI/roberta-large  \
  --task_name ${task_name} \
  --do_train --do_eval \
  --max_seq_length ${ml[$1]} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --train_classifier ${train_classifier} \
  --use_sara ${use_sara} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end True --metric_for_best_model ${metrics[$1]} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0. \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} --wandb_project ${wandb_project} \
  --lora_alpha '768' --lora_dropout 0. --lora_bias none \
  --target_modules ${target_modules} --rank ${rank} \
  --lora_task_type SEQ_CLS  \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --wandb_run_name ${wandb_run_name} \
  --overwrite_output_dir
}

# task_base=('mnli' 'qqp' )

task_base=('mnli' 'cola' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' )

# task_base=('mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola')

for task in "${task_base[@]}"; do
    run $task
done  