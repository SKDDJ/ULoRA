#!/bin/bash

export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
export HF_HOME=/root/.cache/huggingface
export XDG_CACHE_HOME=/root/.cache

# LATEST
# roberta base LoRA Todo: Update epochs
declare -A epochs=(["mnli"]=10 ["sst2"]=10 ["mrpc"]=20 ["cola"]=20 ["qnli"]=10 ["qqp"]=20 ["rte"]=20  ["stsb"]=30 )


# roberta base LoRA
declare -A bs=(["mnli"]=32 ["sst2"]=32 ["mrpc"]=32 ["cola"]=32 ["qnli"]=32 ["qqp"]=32 ["rte"]=32  ["stsb"]=32 )


# roberta base LoRA
declare -A ml=(["mnli"]=128 ["sst2"]=128 ["mrpc"]=512 ["cola"]=128 ["qnli"]=512 ["qqp"]=512 ["rte"]=512  ["stsb"]=512 )


# roberta base LoRA
declare -A lr=(["mnli"]="3e-4" ["sst2"]="4e-4" ["mrpc"]="3e-4" ["cola"]="2e-4" ["qnli"]="2e-4" ["qqp"]="3e-4" ["rte"]="4e-4"  ["stsb"]="2e-4" )

declare -A metrics=(["mnli"]="accuracy" ["mrpc"]="accuracy" ["qnli"]="accuracy" ["qqp"]="accuracy" ["rte"]="accuracy" ["sst2"]="accuracy" ["stsb"]="pearson" ["cola"]="matthews_correlation")

export WANDB_MODE=offline


# 记录脚本开始时间
start_time=$(date +%s)

run(){
  task_name=$1
  learning_rate=${lr[$1]}
  num_train_epochs=${epochs[$1]}
  per_device_train_batch_size=${bs[$1]}
  rank=768
  l_num=12
  seed=42
  use_sara=True
  train_classifier=True
  lora_alpha="768"
  target_modules="query value"
  lora_dropout=0.
  lora_bias=none
  lora_task_type=SEQ_CLS

  init_method=lu # 可选项: 'lu',  'uniform', 'normal', 'constant', 'ones', 'zeros'

  export WANDB_PROJECT=144-LoLDU-${task_name}
  export WANDB_NAME=${init_method}-target-${target_modules}-144-seed-${seed}-bs-${per_device_train_batch_size}
  HF_ENDPOINT=https://hf-mirror.com accelerate launch --num_processes 6 --main_process_port 26691 ./run_glue_sara.py \
  --model_name_or_path FacebookAI/roberta-base  \
  --init_method ${init_method} \
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
  --seed ${seed}  \
  --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} \
  --lora_task_type ${lora_task_type}  \
  --target_modules ${target_modules} --rank ${rank} \
  --lora_bias ${lora_bias} \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --overwrite_output_dir
}

task_base=('mrpc' 'cola' 'rte' 'sst2' 'qnli' 'stsb')
# task_base=('mrpc')

for task in "${task_base[@]}"; do
    run $task
done  



# 记录脚本结束时间
end_time=$(date +%s)

# 计算总耗时
elapsed_time=$((end_time - start_time))

# 将总耗时转换为小时、分钟和秒
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

# 打印并输出总耗时
echo "Total time elapsed: ${hours}h ${minutes}m ${seconds}s"